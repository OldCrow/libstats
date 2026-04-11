#pragma once

// Common distribution includes (consolidates std library and core headers)
#include "libstats/common/distribution_common.h"

// Consolidated distribution platform headers (SIMD, parallel execution, thread pools,
// adaptive caching, etc.)
#include "libstats/common/distribution_platform_common.h"

namespace stats {

/**
 * @brief Thread-safe Beta Distribution for modelling probabilities and proportions.
 *
 * @details The Beta distribution is a continuous probability distribution on [0, 1]
 * that is the conjugate prior for Bernoulli and Binomial likelihoods. It is highly
 * flexible: by varying α and β it can model uniform, U-shaped, J-shaped, and
 * bell-shaped behaviours.
 *
 * @par Mathematical Definition:
 * - PDF: f(x; α, β) = x^(α−1) (1−x)^(β−1) / B(α, β)  for x ∈ (0, 1)
 * - LogPDF: (α−1)·log(x) + (β−1)·log(1−x) − lbeta(α, β)
 * - CDF: I_x(α, β) — regularized incomplete beta function (detail::beta_i)
 * - Quantile: detail::inverse_beta_i  (Newton-Raphson on detail::beta_i)
 * - Parameters: α > 0, β > 0
 * - Support: x ∈ [0, 1]
 *
 * @par Moments:
 * - Mean: α/(α+β)
 * - Variance: αβ / ((α+β)²(α+β+1))
 * - Mode: (α−1)/(α+β−2) for α,β > 1; 0 for α ≤ 1, β > 1; 1 for α > 1, β ≤ 1
 * - Skewness: 2(β−α)√(α+β+1) / ((α+β+2)√(αβ))
 * - Excess kurtosis: 6((α−β)²(α+β+1) − αβ(α+β+2)) / (αβ(α+β+2)(α+β+3))
 *
 * @par Special Cases:
 * - α = β = 1: Uniform(0, 1)
 * - α = β: symmetric distribution (median = mean = 1/2)
 * - α < 1, β < 1: U-shaped (bimodal at 0 and 1)
 * - α > 1, β > 1: unimodal (bell-shaped)
 *
 * @par Batch SIMD:
 * LogPDF requires two VectorOps log calls — one for log(x) and one for log(1−x)
 * — plus one aligned temporary buffer. The eight-step pipeline is:
 *   log(x) → scale by (α−1); compute 1−x → log(1−x) → scale by (β−1);
 *   add both → add log_norm_const.
 * A scalar fixup pass corrects x ≤ 0 and x ≥ 1.
 * Expected speedup: PDF ~8–12x, LogPDF ~12–18x (two vector_log vs one).
 *
 * @par Sampling:
 * X ~ Gamma(α, 1), Y ~ Gamma(β, 1) → X/(X+Y) ~ Beta(α, β).
 *
 * @par Applications:
 * - **Bayesian statistics**: conjugate prior for Bernoulli/Binomial likelihoods
 * - **Proportions and rates**: modelling fractions, success rates
 * - **A/B testing**: posterior over conversion rates
 * - **Order statistics**: distribution of k-th order statistic
 * - **Random variable generation**: via inverse CDF
 *
 * @author libstats Development Team
 * @version 1.0.0
 * @since 1.0.0
 */
class BetaDistribution : public DistributionBase {
   public:
    //==========================================================================
    // 1. CONSTRUCTORS AND DESTRUCTOR
    //==========================================================================

    /**
     * @brief Construct a Beta distribution.
     * @param alpha Shape parameter α (must be positive, default: 1.0)
     * @param beta  Shape parameter β (must be positive, default: 1.0)
     * @throws std::invalid_argument if either parameter is not positive and finite
     *
     * Default (1, 1) is the Uniform(0, 1) distribution.
     * Implementation in .cpp.
     */
    explicit BetaDistribution(double alpha = detail::ONE, double beta = detail::ONE);

    /** @brief Thread-safe copy constructor. Implementation in .cpp. */
    BetaDistribution(const BetaDistribution& other);

    /** @brief Copy assignment operator. Implementation in .cpp. */
    BetaDistribution& operator=(const BetaDistribution& other);

    /** @brief Move constructor. Implementation in .cpp. */
    BetaDistribution(BetaDistribution&& other) noexcept;

    /** @brief Move assignment operator. Implementation in .cpp. */
    BetaDistribution& operator=(BetaDistribution&& other) noexcept;

    /** @brief Destructor — defaulted. */
    ~BetaDistribution() override = default;

    //==========================================================================
    // 2. SAFE FACTORY METHODS (Exception-free construction)
    //==========================================================================

    /**
     * @brief Safely create a Beta distribution without throwing exceptions.
     * @param alpha Shape parameter α (must be positive)
     * @param beta  Shape parameter β (must be positive)
     * @return Result containing a valid BetaDistribution or error info
     */
    [[nodiscard]] static Result<BetaDistribution> create(double alpha = detail::ONE,
                                                         double beta = detail::ONE) noexcept {
        auto validation = validateBetaParameters(alpha, beta);
        if (validation.isError()) {
            return Result<BetaDistribution>::makeError(validation.error_code, validation.message);
        }
        return Result<BetaDistribution>::ok(createUnchecked(alpha, beta));
    }

    //==========================================================================
    // 3. PARAMETER GETTERS AND SETTERS
    //==========================================================================

    /** @brief Get shape parameter α. */
    [[nodiscard]] double getAlpha() const noexcept {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        return alpha_;
    }

    /** @brief Get shape parameter β. */
    [[nodiscard]] double getBeta() const noexcept {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        return beta_;
    }

    /**
     * @brief Set shape parameter α.
     * @throws std::invalid_argument if alpha <= 0
     */
    void setAlpha(double alpha);

    /**
     * @brief Set shape parameter β.
     * @throws std::invalid_argument if beta <= 0
     */
    void setBeta(double beta);

    /**
     * @brief Set both parameters simultaneously.
     * @throws std::invalid_argument if either parameter <= 0
     */
    void setParameters(double alpha, double beta);

    /** @brief Mean = α/(α+β). */
    [[nodiscard]] double getMean() const noexcept override;

    /** @brief Variance = αβ / ((α+β)²(α+β+1)). */
    [[nodiscard]] double getVariance() const noexcept override;

    /** @brief Skewness. */
    [[nodiscard]] double getSkewness() const noexcept override;

    /** @brief Excess kurtosis. */
    [[nodiscard]] double getKurtosis() const noexcept override;

    /** @brief Number of parameters (always 2). */
    [[nodiscard]] int getNumParameters() const noexcept override { return 2; }

    /** @brief Distribution name. */
    [[nodiscard]] std::string getDistributionName() const override { return "BetaDistribution"; }

    /** @brief Beta is continuous. */
    [[nodiscard]] bool isDiscrete() const noexcept override { return false; }

    /** @brief Support lower bound: 0. */
    [[nodiscard]] double getSupportLowerBound() const noexcept override {
        return detail::ZERO_DOUBLE;
    }

    /** @brief Support upper bound: 1. */
    [[nodiscard]] double getSupportUpperBound() const noexcept override { return detail::ONE; }

    //==========================================================================
    // 4. RESULT-BASED SETTERS
    //==========================================================================

    [[nodiscard]] VoidResult trySetAlpha(double alpha) noexcept;
    [[nodiscard]] VoidResult trySetBeta(double beta) noexcept;
    [[nodiscard]] VoidResult trySetParameters(double alpha, double beta) noexcept;
    [[nodiscard]] VoidResult validateCurrentParameters() const noexcept;

    //==========================================================================
    // 5. CORE PROBABILITY METHODS
    //==========================================================================

    /**
     * @brief PDF at x: x^(α−1)(1−x)^(β−1)/B(α,β) for x ∈ (0,1); 0 outside.
     * Computed via exp(LogPDF) for numerical stability.
     */
    [[nodiscard]] double getProbability(double x) const override;

    /**
     * @brief Log-PDF at x: (α−1)·log(x) + (β−1)·log(1−x) − lbeta(α,β).
     * Returns −∞ for x outside (0,1) (or ±∞ at boundaries for α,β < 1).
     */
    [[nodiscard]] double getLogProbability(double x) const noexcept override;

    /**
     * @brief CDF via detail::beta_i(x, α, β) (regularized incomplete beta).
     */
    [[nodiscard]] double getCumulativeProbability(double x) const override;

    /**
     * @brief Quantile via detail::inverse_beta_i(p, α, β).
     * @throws std::invalid_argument if p not in [0, 1]
     */
    [[nodiscard]] double getQuantile(double p) const override;

    /**
     * @brief Generate a single random sample using X/(X+Y) with X~Gamma(α,1), Y~Gamma(β,1).
     */
    [[nodiscard]] double sample(std::mt19937& rng) const override;

    /**
     * @brief Generate n random samples.
     */
    [[nodiscard]] std::vector<double> sample(std::mt19937& rng, size_t n) const override;

    //==========================================================================
    // 6. DISTRIBUTION MANAGEMENT
    //==========================================================================

    /**
     * @brief Fit α and β to data by MLE.
     *
     * Method-of-moments provides closed-form initial estimates:
     *   c = mean*(1−mean)/variance − 1; α₀ = mean*c; β₀ = (1−mean)*c
     * then Newton-Raphson on the two-dimensional score system using detail::digamma.
     *
     * @param values Observed proportions in (0, 1)
     * @throws std::invalid_argument if values is empty or contains values outside (0, 1)
     */
    void fit(const std::vector<double>& values) override;

    /** @brief Reset to default (α = β = 1, Uniform). */
    void reset() noexcept override;

    /** @brief String representation. */
    std::string toString() const override;

    //==========================================================================
    // 12. DISTRIBUTION-SPECIFIC UTILITY METHODS
    //==========================================================================

    /** @brief Entropy of the distribution (uses detail::digamma). */
    [[nodiscard]] double getEntropy() const override;

    /** @brief Mode: (α−1)/(α+β−2) for α,β>1; boundary values otherwise. */
    [[nodiscard]] double getMode() const noexcept;

    /** @brief Median — numerical via getQuantile(0.5). */
    [[nodiscard]] double getMedian() const noexcept { return getQuantile(detail::HALF); }

    /**
     * @brief True if α = β = 1 within tolerance (Uniform(0,1) distribution).
     */
    [[nodiscard]] bool isUniform() const noexcept {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        return isUniform_;
    }

    /**
     * @brief True if α = β within tolerance (symmetric distribution).
     */
    [[nodiscard]] bool isSymmetric() const noexcept {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        return isSymmetric_;
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

    bool operator==(const BetaDistribution& other) const;
    bool operator!=(const BetaDistribution& other) const;

    //==========================================================================
    // 16. FRIEND FUNCTION STREAM OPERATORS
    //==========================================================================

    friend std::istream& operator>>(std::istream& is, stats::BetaDistribution&);
    friend std::ostream& operator<<(std::ostream& os, const stats::BetaDistribution&);

   private:
    //==========================================================================
    // 17. PRIVATE FACTORY METHODS
    //==========================================================================

    static BetaDistribution createUnchecked(double alpha, double beta) noexcept;
    BetaDistribution(double alpha, double beta, bool /*bypassValidation*/) noexcept;

    //==========================================================================
    // 18. PRIVATE BATCH IMPLEMENTATION METHODS
    //==========================================================================

    /**
     * @brief SIMD log-space pipeline for LogPDF.
     *
     * Eight-step VectorOps pipeline using one aligned temporary:
     *   temp   = log(x)                    [vector_log]
     *   temp   = (α−1)·log(x)              [scalar_multiply]
     *   results = x−1                      [scalar_add(values, -1)]
     *   results = 1−x                      [scalar_multiply(results, -1)]
     *   results = log(1−x)                 [vector_log]
     *   results = (β−1)·log(1−x)           [scalar_multiply]
     *   results = (α−1)log(x)+(β−1)log(1-x) [vector_add(temp, results)]
     *   results += log_norm_const          [scalar_add]
     *
     * Scalar fixup: for x ≤ 0 or x ≥ 1, overwrites with scalar result.
     * PDF variant: append vector_exp.
     */
    void getProbabilityBatchUnsafeImpl(const double* values, double* results, std::size_t count,
                                       double log_norm_const, double alpha_minus_one,
                                       double beta_minus_one) const noexcept;

    void getLogProbabilityBatchUnsafeImpl(const double* values, double* results, std::size_t count,
                                          double log_norm_const, double alpha_minus_one,
                                          double beta_minus_one) const noexcept;

    /** @brief CDF kernel — scalar detail::beta_i per element. */
    void getCumulativeProbabilityBatchUnsafeImpl(const double* values, double* results,
                                                 std::size_t count, double alpha,
                                                 double beta) const noexcept;

    //==========================================================================
    // 20. PRIVATE CACHE MANAGEMENT
    //==========================================================================

    void updateCacheUnsafe() const noexcept override;

    //==========================================================================
    // 21. PRIVATE VALIDATION METHODS
    //==========================================================================

    static void validateParameters(double alpha, double beta) {
        if (std::isnan(alpha) || std::isinf(alpha) || alpha <= detail::ZERO_DOUBLE) {
            throw std::invalid_argument("Alpha (shape1) must be a positive finite number");
        }
        if (std::isnan(beta) || std::isinf(beta) || beta <= detail::ZERO_DOUBLE) {
            throw std::invalid_argument("Beta (shape2) must be a positive finite number");
        }
    }

    [[nodiscard]] static VoidResult validateBetaParameters(double alpha, double beta) noexcept {
        if (std::isnan(alpha) || std::isinf(alpha) || alpha <= detail::ZERO_DOUBLE) {
            return VoidResult::makeError(ValidationError::InvalidParameter,
                                         "Alpha (shape1) must be a positive finite number");
        }
        if (std::isnan(beta) || std::isinf(beta) || beta <= detail::ZERO_DOUBLE) {
            return VoidResult::makeError(ValidationError::InvalidParameter,
                                         "Beta (shape2) must be a positive finite number");
        }
        return VoidResult::ok(true);
    }

    //==========================================================================
    // 23. DISTRIBUTION PARAMETERS
    //==========================================================================

    /** @brief Shape parameter α — must be positive. */
    double alpha_{detail::ONE};

    /** @brief Shape parameter β — must be positive. */
    double beta_{detail::ONE};

    /** @brief Atomic copies for lock-free access. */
    mutable std::atomic<double> atomicAlpha_{detail::ONE};
    mutable std::atomic<double> atomicBeta_{detail::ONE};
    mutable std::atomic<bool> atomicParamsValid_{false};

    //==========================================================================
    // 22. PERFORMANCE CACHE
    //==========================================================================

    /** @brief α − 1 */
    mutable double alphaMinus1_{detail::ZERO_DOUBLE};

    /** @brief β − 1 */
    mutable double betaMinus1_{detail::ZERO_DOUBLE};

    /** @brief −lbeta(α, β) — added in log-space to give log_norm_const */
    mutable double logNormConst_{detail::ZERO_DOUBLE};

    /** @brief Cached mean α/(α+β) */
    mutable double mean_{detail::HALF};

    /** @brief Cached variance αβ/((α+β)²(α+β+1)) */
    mutable double variance_{detail::ZERO_DOUBLE};

    /** @brief Cached mode */
    mutable double mode_{detail::ZERO_DOUBLE};

    //==========================================================================
    // 23. OPTIMIZATION FLAGS
    //==========================================================================

    /** @brief True if α = β = 1 (Uniform distribution) within tolerance. */
    mutable bool isUniform_{true};

    /** @brief True if α = β within tolerance (symmetric). */
    mutable bool isSymmetric_{true};

    /** @brief True if α > 1 and β > 1 (unimodal bell shape). */
    mutable bool isUnimodal_{false};

    /** @brief Atomic cache validity flag for lock-free fast path. */
    mutable std::atomic<bool> cacheValidAtomic_{false};
};

}  // namespace stats

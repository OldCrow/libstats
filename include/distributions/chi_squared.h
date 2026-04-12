#pragma once

// Common distribution includes (consolidates std library and core headers)
#include "libstats/common/distribution_common.h"

// Consolidated distribution platform headers (SIMD, parallel execution, thread pools, adaptive
// caching, etc.)
#include "libstats/common/distribution_platform_common.h"

// Chi-squared is a thin delegation wrapper over GammaDistribution
#include "gamma.h"

namespace stats {

/**
 * @brief Thread-safe Chi-Squared Distribution for goodness-of-fit tests and statistical inference.
 *
 * @details The chi-squared distribution is a special case of the Gamma distribution used
 * extensively in hypothesis testing, confidence interval construction, and goodness-of-fit
 * analysis. It arises naturally as the distribution of a sum of squared independent standard
 * normal random variables.
 *
 * @par Mathematical Definition:
 * - PDF: f(x; k) = x^(k/2 - 1) * exp(-x/2) / (2^(k/2) * Γ(k/2))  for x > 0
 * - CDF: F(x; k) = γ(k/2, x/2) / Γ(k/2) (regularized incomplete gamma function)
 * - Parameter: k > 0 (degrees of freedom)
 * - Support: x ∈ [0, ∞)
 * - Mean: k
 * - Variance: 2k
 * - Mode: max(k - 2, 0)
 *
 * @par Relationship to Gamma:
 * χ²(k) = Gamma(α = k/2, β = 1/2) exactly.
 *
 * @par Delegation Design Pattern:
 * ChiSquaredDistribution is implemented as a **delegation wrapper** over GammaDistribution.
 * A private `GammaDistribution gamma_` member is always kept in sync as Gamma(k/2, 0.5).
 * All probability, batch, quantile, and sampling operations are one-line pass-throughs to
 * `gamma_`. This pattern eliminates code duplication while providing a chi-squared-flavoured API
 * and automatically inheriting all future improvements to GammaDistribution (including SIMD
 * batch vectorization).
 *
 * This is a practical demonstration that when a distribution is an exact special case of another,
 * delegation is architecturally superior to re-implementing the same mathematics.
 *
 * @par Thread Safety:
 * - All methods are fully thread-safe using reader-writer locks
 * - `ChiSquaredDistribution::cache_mutex_` and `gamma_::cache_mutex_` are independent mutexes
 *   on distinct objects; acquiring the outer lock and then calling `gamma_`'s setters (which
 *   acquire the inner lock) has no lock-ordering conflict because `gamma_` is private and
 *   never exposed to external code
 *
 * @par Performance:
 * Batch operations (SIMD/parallel) run through GammaDistribution's dispatch infrastructure,
 * which uses the VectorOps log-space pipeline. The expected SIMD profile matches Gamma.
 *
 * @par Usage Examples:
 * @code
 * // Goodness-of-fit test: compare test statistic to chi-squared critical value
 * auto result = ChiSquaredDistribution::create(5.0);  // 5 degrees of freedom
 * if (result.isOk()) {
 *     auto& chi2 = result.value;
 *
 *     double test_statistic = 11.07;
 *     double p_value = 1.0 - chi2.getCumulativeProbability(test_statistic);
 *     // p_value ≈ 0.05 → reject at 5% significance
 *
 *     // Batch computation: evaluate PDF over a grid
 *     std::vector<double> x(1000);
 *     std::vector<double> pdf(1000);
 *     chi2.getProbability(std::span<const double>(x), std::span<double>(pdf));
 * }
 * @endcode
 *
 * @par Applications:
 * - **Goodness-of-fit tests**: Pearson chi-squared test, G-test
 * - **Independence tests**: Contingency table analysis
 * - **Variance inference**: Confidence intervals for normal variance
 * - **Likelihood ratio tests**: Model comparison (test statistic ~ χ²(df))
 * - **Bayesian statistics**: Prior on precision parameters
 *
 * @par Statistical Properties:
 * - Skewness: √(8/k) (right-skewed, approaches 0 as k increases)
 * - Kurtosis: 12/k (excess kurtosis)
 * - Entropy: k/2 + log(2) + log(Γ(k/2)) + (1 - k/2)ψ(k/2)
 * - Moment generating function: (1 - 2t)^(-k/2) for t < 1/2
 *
 * @author libstats Development Team
 * @version 1.1.0
 * @since 1.0.0
 */
class ChiSquaredDistribution : public DistributionBase {
   public:
    //==========================================================================
    // 1. CONSTRUCTORS AND DESTRUCTOR
    //==========================================================================

    /**
     * @brief Constructs a chi-squared distribution with the given degrees of freedom.
     *
     * @param k Degrees of freedom (must be positive, default: 1.0)
     * @throws std::invalid_argument if k is not a positive finite number
     *
     * Implementation in .cpp: validates k and syncs the internal GammaDistribution.
     */
    explicit ChiSquaredDistribution(double k = detail::ONE);

    /**
     * @brief Thread-safe copy constructor.
     * Implementation in .cpp: locks source, copies k_, reconstructs gamma_.
     */
    ChiSquaredDistribution(const ChiSquaredDistribution& other);

    /**
     * @brief Copy assignment operator.
     * Implementation in .cpp: deadlock-safe with std::lock.
     */
    ChiSquaredDistribution& operator=(const ChiSquaredDistribution& other);

    /**
     * @brief Move constructor.
     * Implementation in .cpp: thread-safe move with locking.
     */
    ChiSquaredDistribution(ChiSquaredDistribution&& other) noexcept;

    /**
     * @brief Move assignment operator.
     * Implementation in .cpp: thread-safe move with deadlock prevention.
     */
    ChiSquaredDistribution& operator=(ChiSquaredDistribution&& other) noexcept;

    /**
     * @brief Destructor — explicitly defaulted to satisfy Rule of Five.
     */
    ~ChiSquaredDistribution() override = default;

    //==========================================================================
    // 2. SAFE FACTORY METHODS (Exception-free construction)
    //==========================================================================

    /**
     * @brief Safely create a chi-squared distribution without throwing exceptions.
     *
     * @param k Degrees of freedom (must be positive)
     * @return Result containing either a valid ChiSquaredDistribution or error info
     */
    [[nodiscard]] static Result<ChiSquaredDistribution> create(double k = detail::ONE) noexcept {
        auto validation = validateChiSquaredParameters(k);
        if (validation.isError()) {
            return Result<ChiSquaredDistribution>::makeError(validation.error_code,
                                                             validation.message);
        }
        return Result<ChiSquaredDistribution>::ok(createUnchecked(k));
    }

    //==========================================================================
    // 3. PARAMETER GETTERS AND SETTERS
    //==========================================================================

    /**
     * @brief Get the degrees of freedom k.
     * @return Current degrees of freedom value
     */
    [[nodiscard]] double getK() const noexcept {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        return k_;
    }

    /**
     * @brief Synonym for getK() — matches chi-squared statistical notation.
     * @return Current degrees of freedom value
     */
    [[nodiscard]] double getDegreesOfFreedom() const noexcept { return getK(); }

    /**
     * @brief Set the degrees of freedom.
     * @param k New degrees of freedom (must be positive)
     * @throws std::invalid_argument if k <= 0
     */
    void setK(double k);

    /**
     * @brief Synonym for setK() — matches chi-squared statistical notation.
     * @param k New degrees of freedom (must be positive)
     * @throws std::invalid_argument if k <= 0
     */
    void setDegreesOfFreedom(double k) { setK(k); }

    /**
     * @brief Set parameters (single-parameter alias for setK).
     * @param k New degrees of freedom (must be positive)
     * @throws std::invalid_argument if k <= 0
     */
    void setParameters(double k) { setK(k); }

    /**
     * @brief Get the mean of the distribution.
     * For chi-squared, mean = k.
     */
    [[nodiscard]] double getMean() const noexcept override { return gamma_.getMean(); }

    /**
     * @brief Get the variance of the distribution.
     * For chi-squared, variance = 2k.
     */
    [[nodiscard]] double getVariance() const noexcept override { return gamma_.getVariance(); }

    /**
     * @brief Get the skewness: √(8/k).
     */
    [[nodiscard]] double getSkewness() const noexcept override { return gamma_.getSkewness(); }

    /**
     * @brief Get the excess kurtosis: 12/k.
     */
    [[nodiscard]] double getKurtosis() const noexcept override { return gamma_.getKurtosis(); }

    /**
     * @brief Get the number of parameters (always 1 for chi-squared).
     */
    [[nodiscard]] int getNumParameters() const noexcept override { return 1; }

    /**
     * @brief Get the distribution name.
     */
    [[nodiscard]] std::string getDistributionName() const override {
        return "ChiSquaredDistribution";
    }

    /**
     * @brief Chi-squared is continuous.
     */
    [[nodiscard]] bool isDiscrete() const noexcept override { return false; }

    /**
     * @brief Support lower bound: 0.
     */
    [[nodiscard]] double getSupportLowerBound() const noexcept override {
        return detail::ZERO_DOUBLE;
    }

    /**
     * @brief Support upper bound: +∞.
     */
    [[nodiscard]] double getSupportUpperBound() const noexcept override {
        return std::numeric_limits<double>::infinity();
    }

    //==========================================================================
    // 4. RESULT-BASED SETTERS
    //==========================================================================

    /**
     * @brief Safely set degrees of freedom without throwing exceptions.
     * @param k New degrees of freedom (must be positive)
     * @return VoidResult indicating success or failure
     */
    [[nodiscard]] VoidResult trySetK(double k) noexcept;

    /**
     * @brief Synonym for trySetK().
     */
    [[nodiscard]] VoidResult trySetDegreesOfFreedom(double k) noexcept { return trySetK(k); }

    /**
     * @brief Safely set parameters without throwing exceptions.
     * @param k New degrees of freedom (must be positive)
     * @return VoidResult indicating success or failure
     */
    [[nodiscard]] VoidResult trySetParameters(double k) noexcept { return trySetK(k); }

    /**
     * @brief Check if current parameters are valid.
     */
    [[nodiscard]] VoidResult validateCurrentParameters() const noexcept;

    //==========================================================================
    // 5. CORE PROBABILITY METHODS — delegated to gamma_
    //==========================================================================

    /** @brief PDF at x — delegates to GammaDistribution. */
    [[nodiscard]] double getProbability(double x) const override {
        return gamma_.getProbability(x);
    }

    /** @brief Log-PDF at x — delegates to GammaDistribution. */
    [[nodiscard]] double getLogProbability(double x) const noexcept override {
        return gamma_.getLogProbability(x);
    }

    /** @brief CDF at x — delegates to GammaDistribution. */
    [[nodiscard]] double getCumulativeProbability(double x) const override {
        return gamma_.getCumulativeProbability(x);
    }

    /**
     * @brief Quantile function (inverse CDF) — delegates to GammaDistribution.
     * @param p Probability value in [0, 1]
     * @throws std::invalid_argument if p not in [0, 1]
     */
    [[nodiscard]] double getQuantile(double p) const override { return gamma_.getQuantile(p); }

    /**
     * @brief Generate a single random sample — delegates to GammaDistribution.
     * @param rng Random number generator
     */
    [[nodiscard]] double sample(std::mt19937& rng) const override { return gamma_.sample(rng); }

    /**
     * @brief Generate multiple random samples — delegates to GammaDistribution.
     * @param rng Random number generator
     * @param n Number of samples
     */
    [[nodiscard]] std::vector<double> sample(std::mt19937& rng, size_t n) const override {
        return gamma_.sample(rng, n);
    }

    //==========================================================================
    // 6. DISTRIBUTION MANAGEMENT
    //==========================================================================

    /**
     * @brief Fit k to data using MLE.
     *
     * For chi-squared, MLE is k̂ = sample_mean (since E[X] = k).
     * No iterative solver required.
     *
     * @param values Observed positive data
     * @throws std::invalid_argument if values is empty or contains non-positive values
     */
    void fit(const std::vector<double>& values) override;

    /**
     * @brief Reset to default parameters (k = 1).
     */
    void reset() noexcept override;

    /**
     * @brief String representation of the distribution.
     */
    std::string toString() const override;

    //==========================================================================
    // 12. DISTRIBUTION-SPECIFIC UTILITY METHODS
    //==========================================================================

    /**
     * @brief Entropy of the distribution — delegates to GammaDistribution.
     * H = k/2 + log(2) + log(Γ(k/2)) + (1 - k/2)ψ(k/2)
     */
    [[nodiscard]] double getEntropy() const override { return gamma_.getEntropy(); }

    /**
     * @brief Median — delegates to GammaDistribution (numerical, via quantile).
     */
    [[nodiscard]] double getMedian() const noexcept { return gamma_.getMedian(); }

    /**
     * @brief Mode = max(k - 2, 0) — delegates to GammaDistribution.
     */
    [[nodiscard]] double getMode() const noexcept { return gamma_.getMode(); }

    //==========================================================================
    // 13. SMART AUTO-DISPATCH BATCH OPERATIONS — delegated to gamma_
    //==========================================================================

    /**
     * @brief Batch PDF — delegates SIMD/parallel dispatch to GammaDistribution.
     * @param values Input values
     * @param results Output densities (same size as values)
     * @param hint Performance hint (default: AUTO)
     */
    void getProbability(std::span<const double> values, std::span<double> results,
                        const detail::PerformanceHint& hint = {}) const {
        gamma_.getProbability(values, results, hint);
    }

    /**
     * @brief Batch log-PDF — delegates SIMD/parallel dispatch to GammaDistribution.
     */
    void getLogProbability(std::span<const double> values, std::span<double> results,
                           const detail::PerformanceHint& hint = {}) const {
        gamma_.getLogProbability(values, results, hint);
    }

    /**
     * @brief Batch CDF — delegates SIMD/parallel dispatch to GammaDistribution.
     */
    void getCumulativeProbability(std::span<const double> values, std::span<double> results,
                                  const detail::PerformanceHint& hint = {}) const {
        gamma_.getCumulativeProbability(values, results, hint);
    }

    //==========================================================================
    // 14. EXPLICIT STRATEGY BATCH OPERATIONS — delegated to gamma_
    //==========================================================================

    /** @brief Explicit-strategy batch PDF — delegates to GammaDistribution. */
    void getProbabilityWithStrategy(std::span<const double> values, std::span<double> results,
                                    detail::Strategy strategy) const {
        gamma_.getProbabilityWithStrategy(values, results, strategy);
    }

    /** @brief Explicit-strategy batch log-PDF — delegates to GammaDistribution. */
    void getLogProbabilityWithStrategy(std::span<const double> values, std::span<double> results,
                                       detail::Strategy strategy) const {
        gamma_.getLogProbabilityWithStrategy(values, results, strategy);
    }

    /** @brief Explicit-strategy batch CDF — delegates to GammaDistribution. */
    void getCumulativeProbabilityWithStrategy(std::span<const double> values,
                                              std::span<double> results,
                                              detail::Strategy strategy) const {
        gamma_.getCumulativeProbabilityWithStrategy(values, results, strategy);
    }

    //==========================================================================
    // 15. COMPARISON OPERATORS
    //==========================================================================

    bool operator==(const ChiSquaredDistribution& other) const;
    bool operator!=(const ChiSquaredDistribution& other) const;

    //==========================================================================
    // 16. FRIEND FUNCTION STREAM OPERATORS
    //==========================================================================

    friend std::istream& operator>>(std::istream& is, stats::ChiSquaredDistribution&);
    friend std::ostream& operator<<(std::ostream& os, const stats::ChiSquaredDistribution&);

   private:
    //==========================================================================
    // 17. PRIVATE FACTORY METHODS
    //==========================================================================

    /** @brief Create without parameter validation (for internal use). */
    static ChiSquaredDistribution createUnchecked(double k) noexcept;

    /** @brief Private bypass-validation constructor. */
    ChiSquaredDistribution(double k, bool /*bypassValidation*/) noexcept;

    //==========================================================================
    // 18. PRIVATE BATCH IMPLEMENTATION METHODS
    //==========================================================================

    // Not needed: all batch operations delegate to gamma_.
    // Section retained for template compliance.

    //==========================================================================
    // 19. PRIVATE COMPUTATIONAL METHODS
    //==========================================================================

    // Not needed: all computation delegated to gamma_.
    // Section retained for template compliance.

    //==========================================================================
    // 20. PRIVATE CACHE MANAGEMENT
    //==========================================================================

    /**
     * @brief Sync gamma_ with current k_ and mark cache valid.
     *
     * Called from within a held unique_lock on cache_mutex_.
     * Acquires gamma_'s own mutex internally via trySetAlpha — no lock-ordering
     * conflict because gamma_ is private and never locked by external code.
     */
    void updateCacheUnsafe() const noexcept override;

    //==========================================================================
    // 21. PRIVATE VALIDATION METHODS
    //==========================================================================

    /**
     * @brief Validate chi-squared parameters.
     * @param k Degrees of freedom (must be positive and finite)
     * @throws std::invalid_argument if invalid
     */
    static void validateParameters(double k) {
        if (std::isnan(k) || std::isinf(k) || k <= detail::ZERO_DOUBLE) {
            throw std::invalid_argument("Degrees of freedom k must be a positive finite number");
        }
    }

    /**
     * @brief Result-based parameter validation (no-throw).
     */
    [[nodiscard]] static VoidResult validateChiSquaredParameters(double k) noexcept {
        if (std::isnan(k) || std::isinf(k) || k <= detail::ZERO_DOUBLE) {
            return VoidResult::makeError(ValidationError::InvalidParameter,
                                         "Degrees of freedom k must be a positive finite number");
        }
        return VoidResult::ok(true);
    }

    //==========================================================================
    // 22. PRIVATE UTILITY METHODS
    //==========================================================================

    // Not needed for delegation wrapper.
    // Section retained for template compliance.

    //==========================================================================
    // 23. DISTRIBUTION PARAMETERS
    //==========================================================================

    /** @brief Degrees of freedom k — must be positive. */
    double k_{detail::ONE};

    //==========================================================================
    // 24. PRIVATE DELEGATION MEMBER
    //==========================================================================

    /**
     * @brief Internal Gamma distribution — always maintained as Gamma(k/2, 0.5).
     *
     * This is the heart of the delegation pattern. All probability, batch, quantile,
     * sampling, and statistical computations are forwarded here. This object owns
     * its own thread-safety (cache_mutex_, atomics) which is independent of the
     * outer ChiSquaredDistribution::cache_mutex_.
     *
     * Invariant: gamma_.getAlpha() == k_ / 2  and  gamma_.getBeta() == 0.5
     */
    mutable GammaDistribution gamma_{detail::HALF, detail::HALF};
};

}  // namespace stats

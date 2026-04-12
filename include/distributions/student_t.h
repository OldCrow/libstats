#pragma once

// Common distribution includes (consolidates std library and core headers)
#include "libstats/common/distribution_common.h"

// Consolidated distribution platform headers (SIMD, parallel execution, thread pools,
// adaptive caching, etc.)
#include "libstats/common/distribution_platform_common.h"

namespace stats {

/**
 * @brief Thread-safe Student's t Distribution for robust statistical inference.
 *
 * @details The Student's t distribution is a continuous probability distribution
 * arising from the problem of estimating the mean of a normally distributed population
 * when the sample size is small. It is fundamental in hypothesis testing, confidence
 * interval construction, and Bayesian inference with unknown variance.
 *
 * @par Mathematical Definition:
 * - PDF: f(x; ν) = Γ((ν+1)/2) / (√(νπ) Γ(ν/2)) · (1 + x²/ν)^(−(ν+1)/2)
 * - LogPDF: log_norm_const − (ν+1)/2 · log(1 + x²/ν)
 * - CDF: via regularized incomplete beta function I_{ν/(ν+x²)}(ν/2, 1/2)
 * - Support: x ∈ (−∞, +∞)
 * - Parameter: ν > 0 (degrees of freedom)
 *
 * @par Moments:
 * - Mean: 0 for ν > 1; undefined for ν ≤ 1
 * - Variance: ν/(ν−2) for ν > 2; +∞ for 1 < ν ≤ 2; undefined for ν ≤ 1
 * - Mode and Median: 0 (symmetric around 0 for all ν)
 * - Skewness: 0 for ν > 3; undefined otherwise
 * - Excess kurtosis: 6/(ν−4) for ν > 4; undefined otherwise
 *
 * @par Special Cases:
 * - ν = 1: Cauchy distribution (PDF(0) = 1/π)
 * - ν = 2: PDF(x) = 1/(2(1 + x²/2)^(3/2))
 * - ν → ∞: converges to Normal(0, 1)
 *
 * @par Batch SIMD:
 * LogPDF = log_norm_const − (ν+1)/2 · log(1 + x²/ν) uses a six-step VectorOps
 * pipeline (vector_multiply → scalar_multiply → scalar_add → vector_log →
 * scalar_multiply → scalar_add) with no out-of-support fixup needed since the
 * domain is all of ℝ. PDF adds vector_exp. Expected speedup: ~10–18x on AVX.
 *
 * @par Thread Safety:
 * - All methods are fully thread-safe using reader-writer locks
 * - Atomic parameter access for lock-free fast paths
 *
 * @par Usage Examples:
 * @code
 * // Two-sample t-test: t(ν=18) with ν = n1+n2-2 degrees of freedom
 * auto result = StudentTDistribution::create(18.0);
 * if (result.isOk()) {
 *     auto& t_dist = result.value;
 *
 *     double t_stat = 2.8;
 *     double p_value = 2.0 * (1.0 - t_dist.getCumulativeProbability(std::abs(t_stat)));
 *     // p_value < 0.05: reject H0
 *
 *     // Critical value for two-tailed α=0.05
 *     double t_crit = t_dist.getQuantile(0.975);  // ~2.101 for ν=18
 * }
 * @endcode
 *
 * @par Applications:
 * - **Hypothesis testing**: One-sample, two-sample, paired t-tests
 * - **Confidence intervals**: Mean estimation with unknown variance
 * - **Regression analysis**: Coefficient significance tests
 * - **Bayesian statistics**: Robust likelihood with heavy tails
 * - **Finance**: Fat-tailed return modelling
 *
 * @author libstats Development Team
 * @version 1.1.0
 * @since 1.0.0
 */
class StudentTDistribution : public DistributionBase {
   public:
    //==========================================================================
    // 1. CONSTRUCTORS AND DESTRUCTOR
    //==========================================================================

    /**
     * @brief Construct a Student's t distribution.
     * @param nu Degrees of freedom (must be positive, default: 1.0)
     * @throws std::invalid_argument if nu is not a positive finite number
     *
     * Implementation in .cpp: validates nu and populates the performance cache.
     */
    explicit StudentTDistribution(double nu = detail::ONE);

    /** @brief Thread-safe copy constructor. Implementation in .cpp. */
    StudentTDistribution(const StudentTDistribution& other);

    /** @brief Copy assignment operator. Implementation in .cpp. */
    StudentTDistribution& operator=(const StudentTDistribution& other);

    /** @brief Move constructor. Implementation in .cpp. */
    StudentTDistribution(StudentTDistribution&& other) noexcept;

    /** @brief Move assignment operator. Implementation in .cpp. */
    StudentTDistribution& operator=(StudentTDistribution&& other) noexcept;

    /** @brief Destructor — defaulted. */
    ~StudentTDistribution() override = default;

    //==========================================================================
    // 2. SAFE FACTORY METHODS (Exception-free construction)
    //==========================================================================

    /**
     * @brief Safely create a Student's t distribution without throwing exceptions.
     * @param nu Degrees of freedom (must be positive)
     * @return Result containing either a valid StudentTDistribution or error info
     */
    [[nodiscard]] static Result<StudentTDistribution> create(double nu = detail::ONE) noexcept {
        auto validation = validateStudentTParameters(nu);
        if (validation.isError()) {
            return Result<StudentTDistribution>::makeError(validation.error_code,
                                                           validation.message);
        }
        return Result<StudentTDistribution>::ok(createUnchecked(nu));
    }

    //==========================================================================
    // 3. PARAMETER GETTERS AND SETTERS
    //==========================================================================

    /** @brief Get the degrees of freedom ν. */
    [[nodiscard]] double getNu() const noexcept {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        return nu_;
    }

    /** @brief Synonym for getNu(). */
    [[nodiscard]] double getDegreesOfFreedom() const noexcept { return getNu(); }

    /**
     * @brief Set the degrees of freedom.
     * @throws std::invalid_argument if nu <= 0
     */
    void setNu(double nu);

    /** @brief Synonym for setNu(). */
    void setDegreesOfFreedom(double nu) { setNu(nu); }

    /** @brief Single-parameter alias for setNu(). */
    void setParameters(double nu) { setNu(nu); }

    /** @brief Mean = 0 for ν > 1; NaN otherwise. */
    [[nodiscard]] double getMean() const noexcept override;

    /** @brief Variance = ν/(ν−2) for ν > 2; +∞ for 1 < ν ≤ 2; NaN for ν ≤ 1. */
    [[nodiscard]] double getVariance() const noexcept override;

    /** @brief Skewness = 0 for ν > 3; NaN otherwise. */
    [[nodiscard]] double getSkewness() const noexcept override;

    /** @brief Excess kurtosis = 6/(ν−4) for ν > 4; NaN otherwise. */
    [[nodiscard]] double getKurtosis() const noexcept override;

    /** @brief Number of parameters (always 1). */
    [[nodiscard]] int getNumParameters() const noexcept override { return 1; }

    /** @brief Distribution name. */
    [[nodiscard]] std::string getDistributionName() const override {
        return "StudentTDistribution";
    }

    /** @brief Student's t is continuous. */
    [[nodiscard]] bool isDiscrete() const noexcept override { return false; }

    /** @brief Support lower bound: −∞. */
    [[nodiscard]] double getSupportLowerBound() const noexcept override {
        return -std::numeric_limits<double>::infinity();
    }

    /** @brief Support upper bound: +∞. */
    [[nodiscard]] double getSupportUpperBound() const noexcept override {
        return std::numeric_limits<double>::infinity();
    }

    //==========================================================================
    // 4. RESULT-BASED SETTERS
    //==========================================================================

    /** @brief Set ν without throwing exceptions. */
    [[nodiscard]] VoidResult trySetNu(double nu) noexcept;

    /** @brief Synonym for trySetNu(). */
    [[nodiscard]] VoidResult trySetDegreesOfFreedom(double nu) noexcept { return trySetNu(nu); }

    /** @brief Single-parameter alias for trySetNu(). */
    [[nodiscard]] VoidResult trySetParameters(double nu) noexcept { return trySetNu(nu); }

    /** @brief Check if current parameters are valid. */
    [[nodiscard]] VoidResult validateCurrentParameters() const noexcept;

    //==========================================================================
    // 5. CORE PROBABILITY METHODS
    //==========================================================================

    /**
     * @brief PDF at x: Γ((ν+1)/2)/(√(νπ)·Γ(ν/2)) · (1+x²/ν)^(−(ν+1)/2)
     * Uses log-space computation for numerical stability.
     */
    [[nodiscard]] double getProbability(double x) const override;

    /**
     * @brief Log-PDF at x: log_norm_const − (ν+1)/2 · log(1+x²/ν)
     */
    [[nodiscard]] double getLogProbability(double x) const noexcept override;

    /**
     * @brief CDF via detail::t_cdf(x, ν) (regularized incomplete beta).
     */
    [[nodiscard]] double getCumulativeProbability(double x) const override;

    /**
     * @brief Quantile via detail::inverse_t_cdf(p, ν) (Newton-Raphson).
     * @throws std::invalid_argument if p not in [0, 1]
     */
    [[nodiscard]] double getQuantile(double p) const override;

    /**
     * @brief Generate a single random sample using the Z/√(χ²(ν)/ν) identity.
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
     * @brief Fit ν to data by Newton-Raphson on the MLE score equation.
     *
     * The MLE score for ν is:
     *   ∂L/∂ν = n·[ψ((ν+1)/2) − ψ(ν/2) − 1/ν] − Σlog(1+xᵢ²/ν) + ((ν+1)/ν)·Σxᵢ²/(ν+xᵢ²)
     *
     * Method-of-moments (excess kurtosis) provides the initial estimate when ν > 4;
     * otherwise ν is initialised to 5.
     *
     * @param values Observed data (any finite real values)
     * @throws std::invalid_argument if values is empty or contains non-finite entries
     */
    void fit(const std::vector<double>& values) override;

    /** @brief Reset to default parameters (ν = 1, Cauchy distribution). */
    void reset() noexcept override;

    /** @brief String representation. */
    std::string toString() const override;

    //==========================================================================
    // 12. DISTRIBUTION-SPECIFIC UTILITY METHODS
    //==========================================================================

    /** @brief Entropy of the distribution. */
    [[nodiscard]] double getEntropy() const override;

    /**
     * @brief Median (always 0 by symmetry).
     */
    [[nodiscard]] double getMedian() const noexcept { return detail::ZERO_DOUBLE; }

    /**
     * @brief Mode (always 0).
     */
    [[nodiscard]] double getMode() const noexcept { return detail::ZERO_DOUBLE; }

    /**
     * @brief True if ν = 1 within tolerance (Cauchy distribution).
     */
    [[nodiscard]] bool isCauchy() const noexcept {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        return isCauchy_;
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

    bool operator==(const StudentTDistribution& other) const;
    bool operator!=(const StudentTDistribution& other) const;

    //==========================================================================
    // 16. FRIEND FUNCTION STREAM OPERATORS
    //==========================================================================

    friend std::istream& operator>>(std::istream& is, stats::StudentTDistribution&);
    friend std::ostream& operator<<(std::ostream& os, const stats::StudentTDistribution&);

   private:
    //==========================================================================
    // 17. PRIVATE FACTORY METHODS
    //==========================================================================

    static StudentTDistribution createUnchecked(double nu) noexcept;
    StudentTDistribution(double nu, bool /*bypassValidation*/) noexcept;

    //==========================================================================
    // 18. PRIVATE BATCH IMPLEMENTATION METHODS
    //==========================================================================

    /**
     * @brief SIMD log-space PDF kernel.
     *
     * Six-step VectorOps pipeline:
     *   x² → x²/ν → 1+x²/ν → log(1+x²/ν) → −(ν+1)/2·log(…) → +log_norm_const
     * No out-of-support fixup needed: domain is all of ℝ.
     * PDF variant appends vector_exp.
     */
    void getProbabilityBatchUnsafeImpl(const double* values, double* results, std::size_t count,
                                       double log_norm_const, double neg_half_nu_plus_one,
                                       double inv_nu) const noexcept;

    void getLogProbabilityBatchUnsafeImpl(const double* values, double* results, std::size_t count,
                                          double log_norm_const, double neg_half_nu_plus_one,
                                          double inv_nu) const noexcept;

    /** @brief CDF kernel — scalar detail::t_cdf per element. */
    void getCumulativeProbabilityBatchUnsafeImpl(const double* values, double* results,
                                                 std::size_t count, double nu) const noexcept;

    //==========================================================================
    // 19. PRIVATE COMPUTATIONAL METHODS
    //==========================================================================

    // computeDigamma removed: now detail::digamma in math_utils.h

    //==========================================================================
    // 20. PRIVATE CACHE MANAGEMENT
    //==========================================================================

    void updateCacheUnsafe() const noexcept override;

    //==========================================================================
    // 21. PRIVATE VALIDATION METHODS
    //==========================================================================

    static void validateParameters(double nu) {
        if (std::isnan(nu) || std::isinf(nu) || nu <= detail::ZERO_DOUBLE) {
            throw std::invalid_argument("Degrees of freedom nu must be a positive finite number");
        }
    }

    [[nodiscard]] static VoidResult validateStudentTParameters(double nu) noexcept {
        if (std::isnan(nu) || std::isinf(nu) || nu <= detail::ZERO_DOUBLE) {
            return VoidResult::makeError(ValidationError::InvalidParameter,
                                         "Degrees of freedom nu must be a positive finite number");
        }
        return VoidResult::ok(true);
    }

    //==========================================================================
    // 23. DISTRIBUTION PARAMETERS
    //==========================================================================

    /** @brief Degrees of freedom ν — must be positive. */
    double nu_{detail::ONE};

    /** @brief Atomic copy for lock-free access. */
    mutable std::atomic<double> atomicNu_{detail::ONE};
    mutable std::atomic<bool> atomicParamsValid_{false};

    //==========================================================================
    // 22. PERFORMANCE CACHE
    //==========================================================================

    /** @brief ν/2 */
    mutable double halfNu_{detail::HALF};

    /** @brief (ν+1)/2 */
    mutable double halfNuPlusOne_{detail::ONE};

    /** @brief −(ν+1)/2  — multiplier in the log-space pipeline */
    mutable double negHalfNuPlusOne_{-detail::ONE};

    /** @brief 1/ν — divisor inside log(1 + x²/ν) */
    mutable double invNu_{detail::ONE};

    /** @brief lgamma((ν+1)/2) − 0.5·log(νπ) − lgamma(ν/2) */
    mutable double logNormConst_{detail::ZERO_DOUBLE};

    /** @brief Cached variance: ν/(ν−2) for ν>2, else ∞ or NaN */
    mutable double variance_{std::numeric_limits<double>::quiet_NaN()};

    /** @brief Cached excess kurtosis: 6/(ν−4) for ν>4 */
    mutable double kurtosis_{std::numeric_limits<double>::quiet_NaN()};

    //==========================================================================
    // 23. OPTIMIZATION FLAGS
    //==========================================================================

    /** @brief True if ν = 1 (Cauchy distribution) within tolerance. */
    mutable bool isCauchy_{true};

    /** @brief True if ν > 1 (mean is defined). */
    mutable bool isMeanDefined_{false};

    /** @brief True if ν > 2 (variance is defined and finite). */
    mutable bool isVarianceDefined_{false};

    /** @brief Atomic cache validity flag for lock-free fast path. */
    mutable std::atomic<bool> cacheValidAtomic_{false};
};

}  // namespace stats

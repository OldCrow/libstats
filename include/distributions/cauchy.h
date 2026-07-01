#pragma once

#include "libstats/common/distribution_common.h"

// Cauchy is a thin delegation wrapper over StudentTDistribution(ν=1).
#include "student_t.h"

namespace stats {

/**
 * @brief Thread-safe Cauchy Distribution — heavy-tailed location-scale distribution.
 *
 * @details The Cauchy distribution is the canonical example of a distribution with no
 * finite moments (mean, variance, skewness, and kurtosis are all undefined).  It arises
 * as the ratio of two independent standard normal variables, as the stable distribution
 * with index α=1, and naturally in physics as a resonance line shape (Lorentzian).
 *
 * @par Mathematical Definition:
 * - PDF:    f(x; x₀, γ) = 1 / (πγ(1 + ((x−x₀)/γ)²))   for x ∈ ℝ
 * - LogPDF: −log(πγ) − log(1 + ((x−x₀)/γ)²)
 * - CDF:    0.5 + atan((x−x₀)/γ)/π
 * - Quantile: x₀ + γ·tan(π(p − 0.5))
 * - Parameters: x₀ ∈ ℝ (location), γ > 0 (scale)
 * - Support: x ∈ (−∞, +∞)
 *
 * @par Moments:
 * - Mean, Variance, Skewness, Kurtosis: all undefined (return NaN)
 * - Median = Mode = x₀
 * - Entropy: log(4πγ)  (nats)
 *
 * @par Relationship to Student's t:
 * Cauchy(x₀, γ) ≡ x₀ + γ · StudentT(ν=1).
 * If Z ~ StudentT(1), then x₀ + γZ ~ Cauchy(x₀, γ).
 *
 * @par Delegation Design Pattern:
 * CauchyDistribution delegates all probability computation to a private
 * `StudentTDistribution student_t_{1.0}` member, which is fixed at ν=1 and
 * never needs to be updated when x₀ or γ change.  The location-scale transform
 * is applied to inputs before delegation (z = (x − x₀)/γ) and outputs are
 * scaled accordingly (PDF: ÷γ; LogPDF: −log γ; CDF/Quantile: no scaling).
 * This pattern automatically inherits all future improvements to StudentT
 * (including SIMD batch vectorisation).
 *
 * @par Thread Safety:
 * - All methods are fully thread-safe using reader-writer locks.
 * - `CauchyDistribution::cache_mutex_` and `student_t_::cache_mutex_` are
 *   independent mutexes on distinct objects; acquiring the outer lock and then
 *   calling the `student_t_` batch methods (which acquire the inner lock) has
 *   no lock-ordering conflict because `student_t_` is private and never exposed.
 * - CauchyDistribution has `atomicX0_`/`atomicGamma_`/`atomicParamsValid_` for
 *   lock-free atomic parameter reads.
 *
 * @par Batch Operations:
 * The three batch span overloads allocate a temporary aligned buffer z of size n,
 * transform the inputs (z[i] = (x[i] − x₀)/γ), then call StudentT's batch method
 * on z.  StudentT's own auto-dispatch handles SIMD/parallel routing internally.
 *
 * @par MLE:
 * - Seed: x₀⁽⁰⁾ = median(data), γ⁽⁰⁾ = IQR/2
 * - Iterate 20 Fisher-scoring steps on the Cauchy score equations:
 *     ∂L/∂x₀ = Σᵢ 2dᵢ/(γ²+dᵢ²),   ∂L/∂γ = Σᵢ (−1/γ + 2dᵢ²/(γ(γ²+dᵢ²)))
 *   using expected Fisher information per observation: I(x₀) = I(γ) = 1/(2γ²).
 *
 * @author libstats Development Team
 * @version 2.0.0
 * @since 2.0.0
 */
class CauchyDistribution : public DistributionBase {
   public:
    // Dispatch metadata
    static constexpr detail::DistributionType kDistributionType =
        detail::DistributionType::CAUCHY;
    static constexpr bool kIsDiscrete = false;

   public:
    //==========================================================================
    // 1. CONSTRUCTORS AND DESTRUCTOR
    //==========================================================================

    /**
     * @brief Construct a Cauchy distribution.
     * @param x0    Location parameter (must be finite, default 0)
     * @param gamma Scale parameter (must be positive and finite, default 1)
     * @throws std::invalid_argument if x0 is not finite or gamma ≤ 0
     */
    explicit CauchyDistribution(double x0 = detail::ZERO_DOUBLE, double gamma = detail::ONE);

    /** @brief Thread-safe copy constructor. */
    CauchyDistribution(const CauchyDistribution& other);

    /** @brief Copy assignment operator. */
    CauchyDistribution& operator=(const CauchyDistribution& other);

    /** @brief Move constructor. */
    CauchyDistribution(CauchyDistribution&& other) noexcept;

    /** @brief Move assignment operator. */
    CauchyDistribution& operator=(CauchyDistribution&& other) noexcept;

    /** @brief Destructor — defaulted. */
    ~CauchyDistribution() override = default;

    //==========================================================================
    // 2. SAFE FACTORY METHODS (Exception-free construction)
    //==========================================================================

    /**
     * @brief Safely create a Cauchy distribution without throwing exceptions.
     * @param x0    Location parameter (must be finite)
     * @param gamma Scale parameter (must be positive and finite)
     * @return Result containing a valid CauchyDistribution or error info
     */
    [[nodiscard]] static Result<CauchyDistribution> create(double x0    = detail::ZERO_DOUBLE,
                                                           double gamma = detail::ONE) {
        auto v = validateCauchyParameters(x0, gamma);
        if (v.isError())
            return Result<CauchyDistribution>::makeError(v.errorCode(), v.message());
        return Result<CauchyDistribution>::ok(createUnchecked(x0, gamma));
    }

    //==========================================================================
    // 3. PARAMETER GETTERS AND SETTERS
    //==========================================================================

    /** @brief Get location parameter x₀. */
    [[nodiscard]] double getX0() const noexcept {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        return x0_;
    }

    /** @brief Get scale parameter γ. */
    [[nodiscard]] double getGamma() const noexcept {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        return gamma_;
    }

    /** @brief Lock-free atomic getter for x₀. */
    [[nodiscard]] double getX0Atomic() const noexcept;

    /** @brief Lock-free atomic getter for γ. */
    [[nodiscard]] double getGammaAtomic() const noexcept;

    /**
     * @brief Set location parameter x₀.
     * @throws std::invalid_argument if x0 is not finite
     */
    void setX0(double x0);

    /**
     * @brief Set scale parameter γ.
     * @throws std::invalid_argument if gamma ≤ 0
     */
    void setGamma(double gamma);

    /** @brief Set both parameters simultaneously. */
    void setParameters(double x0, double gamma);

    /** @brief Mean is undefined for Cauchy — returns NaN. */
    [[nodiscard]] double getMean() const noexcept override {
        return std::numeric_limits<double>::quiet_NaN();
    }

    /** @brief Variance is undefined for Cauchy — returns NaN. */
    [[nodiscard]] double getVariance() const noexcept override {
        return std::numeric_limits<double>::quiet_NaN();
    }

    /** @brief Skewness is undefined for Cauchy — returns NaN. */
    [[nodiscard]] double getSkewness() const noexcept override {
        return std::numeric_limits<double>::quiet_NaN();
    }

    /** @brief Kurtosis is undefined for Cauchy — returns NaN. */
    [[nodiscard]] double getKurtosis() const noexcept override {
        return std::numeric_limits<double>::quiet_NaN();
    }

    /** @brief Number of parameters (always 2). */
    [[nodiscard]] int getNumParameters() const noexcept override { return 2; }

    /** @brief Distribution name. */
    [[nodiscard]] std::string_view getDistributionName() const noexcept override {
        return "Cauchy";
    }

    /** @brief Cauchy is continuous. */
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

    [[nodiscard]] VoidResult trySetX0(double x0) noexcept;
    [[nodiscard]] VoidResult trySetGamma(double gamma) noexcept;
    [[nodiscard]] VoidResult trySetParameters(double x0, double gamma) noexcept;
    [[nodiscard]] VoidResult validateCurrentParameters() const noexcept;

    //==========================================================================
    // 5. CORE PROBABILITY METHODS — delegates to StudentT(ν=1) via z=(x−x₀)/γ
    //==========================================================================

    /**
     * @brief PDF: 1/(πγ(1+((x−x₀)/γ)²)) = student_t_(1).PDF((x−x₀)/γ) / γ.
     * Returns NaN for NaN input.
     */
    [[nodiscard]] double getProbability(double x) const override;

    /**
     * @brief LogPDF: −log(πγ) − log(1+((x−x₀)/γ)²) = student_t_(1).LogPDF((x−x₀)/γ) − log γ.
     * Returns NaN for NaN input.
     */
    [[nodiscard]] double getLogProbability(double x) const override;

    /**
     * @brief CDF: 0.5 + atan((x−x₀)/γ)/π = student_t_(1).CDF((x−x₀)/γ).
     * Returns NaN for NaN input; 0/1 for ±∞.
     */
    [[nodiscard]] double getCumulativeProbability(double x) const override;

    /**
     * @brief Quantile (inverse CDF): x₀ + γ·tan(π(p−0.5)).
     * Closed form — no iterative solver.
     * @throws std::invalid_argument if p not in [0, 1]
     */
    [[nodiscard]] double getQuantile(double p) const override;

    /** @brief Single random sample via x₀ + γ·StudentT(1).sample(). */
    [[nodiscard]] double sample(std::mt19937& rng) const override;

    /** @brief Generate n random samples. */
    [[nodiscard]] std::vector<double> sample(std::mt19937& rng, size_t n) const override;

    //==========================================================================
    // 6. DISTRIBUTION MANAGEMENT
    //==========================================================================

    /**
     * @brief Fit x₀ and γ by MLE.
     *
     * Algorithm:
     *   1. Seed x₀⁽⁰⁾ = median(data), γ⁽⁰⁾ = IQR/2.
     *   2. 20 Fisher-scoring iterations on the Cauchy score equations.
     *
     * @param values Observed data (must be finite, non-empty)
     * @throws std::invalid_argument if values is empty or contains non-finite values
     */
    void fit(const std::vector<double>& values) override;

    /** @brief Parallel batch fitting across multiple independent datasets. */
    static void parallelBatchFit(const std::vector<std::vector<double>>& datasets,
                                 std::vector<CauchyDistribution>& results);

    /** @brief Reset to default (x₀ = 0, γ = 1). */
    void reset() noexcept override;

    /** @brief String representation. */
    std::string toString() const override;

    //==========================================================================
    // 12. DISTRIBUTION-SPECIFIC UTILITY METHODS
    //==========================================================================

    /** @brief Mode = x₀ (Cauchy has a unique mode at the location parameter). */
    [[nodiscard]] double getMode() const noexcept {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        return x0_;
    }

    /** @brief Median = x₀ (CDF(x₀) = 0.5 by symmetry). */
    [[nodiscard]] double getMedian() const noexcept override {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        return x0_;
    }

    /**
     * @brief Entropy = log(4πγ)  (nats).
     * Independent of x₀; increasing in γ.
     */
    [[nodiscard]] double getEntropy() const override;

    /** @brief True if this is the standard Cauchy (x₀=0, γ=1). */
    [[nodiscard]] bool isStandard() const noexcept {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        return std::fabs(x0_) < detail::DEFAULT_TOLERANCE &&
               std::fabs(gamma_ - detail::ONE) < detail::DEFAULT_TOLERANCE;
    }

    //==========================================================================
    // 13. SMART AUTO-DISPATCH BATCH OPERATIONS
    // Each method transforms inputs z=(x−x₀)/γ, delegates to StudentT(1)'s
    // auto-dispatch batch, then scales output (PDF: ×1/γ; LogPDF: −logγ; CDF: none).
    //==========================================================================

    /**
     * @brief Batch PDF — delegates to StudentT(1) batch after input transform.
     * PDF_Cauchy(x) = PDF_StudentT(1)((x−x₀)/γ) / γ.
     */
    void getProbability(std::span<const double> values, std::span<double> results,
                        const detail::PerformanceHint& hint = {}) const;

    /**
     * @brief Batch log-PDF — delegates to StudentT(1) batch after input transform.
     * LogPDF_Cauchy(x) = LogPDF_StudentT(1)((x−x₀)/γ) − log γ.
     */
    void getLogProbability(std::span<const double> values, std::span<double> results,
                           const detail::PerformanceHint& hint = {}) const;

    /**
     * @brief Batch CDF — delegates to StudentT(1) batch after input transform.
     * CDF_Cauchy(x) = CDF_StudentT(1)((x−x₀)/γ).
     */
    void getCumulativeProbability(std::span<const double> values, std::span<double> results,
                                  const detail::PerformanceHint& hint = {}) const;

    //==========================================================================
    // 15. COMPARISON OPERATORS
    //==========================================================================

    bool operator==(const CauchyDistribution& other) const;
    bool operator!=(const CauchyDistribution& other) const;

    //==========================================================================
    // 16. FRIEND FUNCTION STREAM OPERATORS
    //==========================================================================

    friend std::istream& operator>>(std::istream& is, stats::CauchyDistribution&);
    friend std::ostream& operator<<(std::ostream& os, const stats::CauchyDistribution&);

   private:
    //==========================================================================
    // 17. PRIVATE FACTORY METHODS
    //==========================================================================

    static CauchyDistribution createUnchecked(double x0, double gamma) noexcept;
    CauchyDistribution(double x0, double gamma, bool /*bypassValidation*/) noexcept;

    //==========================================================================
    // 20. PRIVATE CACHE MANAGEMENT
    //==========================================================================

    /**
     * @brief Recompute derived cache from x0_ / gamma_.
     *
     * CauchyDistribution's own cache: inv_gamma_ = 1/γ, log_gamma_ = log(γ).
     * The student_t_ delegate is fixed at ν=1 and never needs updating.
     * Called from within a held unique_lock on cache_mutex_.
     */
    void updateCacheUnsafe() const noexcept override;

    //==========================================================================
    // 21. PRIVATE VALIDATION METHODS
    //==========================================================================

    // AR-5: delegate to the free function in error_handling.h — no duplicate logic.
    static void validateParameters(double x0, double gamma) {
        auto v = ::stats::validateCauchyParameters(x0, gamma);
        if (v.isError()) throw std::invalid_argument(v.message());
    }

    //==========================================================================
    // 23. DISTRIBUTION PARAMETERS
    //==========================================================================

    /** @brief Location parameter x₀. */
    double x0_{detail::ZERO_DOUBLE};

    /** @brief Scale parameter γ — must be positive. */
    double gamma_{detail::ONE};

    /** @brief Atomic copies for lock-free parameter access. */
    mutable std::atomic<double> atomicX0_{detail::ZERO_DOUBLE};
    mutable std::atomic<double> atomicGamma_{detail::ONE};
    mutable std::atomic<bool>   atomicParamsValid_{false};

    //==========================================================================
    // 24. PERFORMANCE CACHE
    //==========================================================================

    /** @brief 1/γ — used as input transform scale and PDF output scale. */
    mutable double inv_gamma_{detail::ONE};

    /** @brief log(γ) — subtracted from StudentT LogPDF to get Cauchy LogPDF. */
    mutable double log_gamma_{detail::ZERO_DOUBLE};

    //==========================================================================
    // 25. PRIVATE DELEGATION MEMBER
    //==========================================================================

    /**
     * @brief Internal StudentT(ν=1) delegate — always fixed at ν=1.
     *
     * All probability, batch, quantile, and sampling computations route through
     * here with input transform z=(x−x₀)/γ. Unlike ChiSquared→Gamma, the
     * delegate parameters never change; only the Cauchy-level x₀ and γ vary.
     *
     * This member owns its own thread-safety (cache_mutex_, atomics) independent
     * of the outer CauchyDistribution::cache_mutex_.
     */
    mutable StudentTDistribution student_t_{detail::ONE};
};

}  // namespace stats

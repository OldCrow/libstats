#pragma once

#include "libstats/common/distribution_common.h"
#include "libstats/common/distribution_platform_common.h"

namespace stats {

/**
 * @brief Thread-safe Laplace Distribution (double-exponential).
 *
 * @details The Laplace distribution is the continuous limit of a symmetric
 * geometric distribution.  Its heavier-than-Gaussian tails and closed-form
 * median MLE make it popular in signal processing, finance (asset returns),
 * and Bayesian regression (Lasso prior).
 *
 * @par Mathematical Definition:
 * - PDF:    f(x; μ, b) = (1/(2b))·exp(−|x−μ|/b)  for x ∈ ℝ
 * - LogPDF: −log(2b) − |x−μ|/b
 * - CDF:    0.5·exp((x−μ)/b)              for x ≤ μ
 *           1 − 0.5·exp(−(x−μ)/b)         for x > μ
 * - Quantile: μ + b·sign(p−0.5)·log(1 − 2|p−0.5|)
 *             equivalently: μ + b·log(2p) for p < 0.5
 *                           μ − b·log(2(1−p)) for p > 0.5
 * - Parameters: μ ∈ ℝ (location), b > 0 (scale)
 * - Support: x ∈ (−∞, +∞)
 *
 * @par Moments:
 * - Mean = Median = Mode = μ
 * - Variance: 2b²
 * - Skewness: 0 (symmetric)
 * - Excess kurtosis: 3
 * - Entropy: 1 + log(2b)  (nats)
 *
 * @par Batch SIMD (LogPDF and PDF):
 * Four-step pipeline using a single temp buffer:
 *   Step 1: tmp = x − μ          [scalar_add(values, −μ, tmp)]
 *   Step 2: tmp = |x − μ|        [scalar fabs loop, auto-vectorised by the compiler]
 *   Step 3: tmp = −|x−μ|/b       [scalar_multiply(tmp, −1/b, tmp)]
 *   Step 4: LogPDF = tmp − log(2b) [scalar_add(tmp, −log(2b), results)]
 *   (PDF: append vector_exp)
 * CDF uses a scalar piecewise loop (signed branch prevents clean vectorisation;
 * parallel execution provides throughput for large batches).
 *
 * @par MLE:
 * - μ̂ = median(data)               O(n log n) sort
 * - b̂ = (1/n)·Σ|xᵢ − μ̂|           O(n) after sort
 * Both estimates are closed-form with no iterative solver required.
 *
 * @author libstats Development Team
 * @version 2.0.0
 * @since 2.0.0
 */
class LaplaceDistribution : public DistributionBase {
   public:
    // Dispatch metadata
    static constexpr detail::DistributionType kDistributionType =
        detail::DistributionType::LAPLACE;
    static constexpr bool kIsDiscrete = false;

   public:
    //==========================================================================
    // 1. CONSTRUCTORS AND DESTRUCTOR
    //==========================================================================

    /**
     * @brief Construct a Laplace distribution.
     * @param mu Location parameter μ (must be finite, default 0)
     * @param b  Scale parameter b (must be positive and finite, default 1)
     * @throws std::invalid_argument if μ is not finite or b ≤ 0
     */
    explicit LaplaceDistribution(double mu = detail::ZERO_DOUBLE, double b = detail::ONE);

    /** @brief Thread-safe copy constructor. */
    LaplaceDistribution(const LaplaceDistribution& other);

    /** @brief Copy assignment operator. */
    LaplaceDistribution& operator=(const LaplaceDistribution& other);

    /** @brief Move constructor. */
    LaplaceDistribution(LaplaceDistribution&& other) noexcept;

    /** @brief Move assignment operator. */
    LaplaceDistribution& operator=(LaplaceDistribution&& other) noexcept;

    /** @brief Destructor — defaulted. */
    ~LaplaceDistribution() override = default;

    //==========================================================================
    // 2. SAFE FACTORY METHODS (Exception-free construction)
    //==========================================================================

    /**
     * @brief Safely create a Laplace distribution without throwing exceptions.
     * @param mu Location parameter (must be finite)
     * @param b  Scale parameter (must be positive and finite)
     * @return Result containing a valid LaplaceDistribution or error info
     */
    [[nodiscard]] static Result<LaplaceDistribution> create(double mu = detail::ZERO_DOUBLE,
                                                            double b  = detail::ONE) {
        auto v = validateLaplaceParameters(mu, b);
        if (v.isError())
            return Result<LaplaceDistribution>::makeError(v.error_code, v.message);
        return Result<LaplaceDistribution>::ok(createUnchecked(mu, b));
    }

    //==========================================================================
    // 3. PARAMETER GETTERS AND SETTERS
    //==========================================================================

    /** @brief Get location parameter μ. */
    [[nodiscard]] double getMu() const noexcept {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        return mu_;
    }

    /** @brief Get scale parameter b. */
    [[nodiscard]] double getB() const noexcept {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        return b_;
    }

    /** @brief Lock-free atomic getter for μ. */
    [[nodiscard]] double getMuAtomic() const noexcept;

    /** @brief Lock-free atomic getter for b. */
    [[nodiscard]] double getBAtomic() const noexcept;

    /**
     * @brief Set location parameter μ.
     * @throws std::invalid_argument if μ is not finite
     */
    void setMu(double mu);

    /**
     * @brief Set scale parameter b.
     * @throws std::invalid_argument if b ≤ 0
     */
    void setB(double b);

    /** @brief Set both parameters simultaneously. */
    void setParameters(double mu, double b);

    /** @brief Mean = Median = Mode = μ. */
    [[nodiscard]] double getMean() const override {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        return mu_;
    }

    /** @brief Variance = 2b². */
    [[nodiscard]] double getVariance() const override {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        return detail::TWO * b_ * b_;
    }

    /** @brief Skewness = 0 (symmetric). */
    [[nodiscard]] double getSkewness() const noexcept override {
        return detail::ZERO_DOUBLE;
    }

    /** @brief Excess kurtosis = 3. */
    [[nodiscard]] double getKurtosis() const noexcept override {
        return detail::THREE;
    }

    /** @brief Number of parameters (always 2). */
    [[nodiscard]] int getNumParameters() const noexcept override { return 2; }

    /** @brief Distribution name. */
    [[nodiscard]] std::string_view getDistributionName() const noexcept override {
        return "LaplaceDistribution";
    }

    /** @brief Laplace is continuous. */
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

    [[nodiscard]] VoidResult trySetMu(double mu) noexcept;
    [[nodiscard]] VoidResult trySetB(double b) noexcept;
    [[nodiscard]] VoidResult trySetParameters(double mu, double b) noexcept;
    [[nodiscard]] VoidResult validateCurrentParameters() const noexcept;

    //==========================================================================
    // 5. CORE PROBABILITY METHODS
    //==========================================================================

    /**
     * @brief PDF: (1/(2b))·exp(−|x−μ|/b).
     * Computed via exp(LogPDF) to reuse cached constants.
     */
    [[nodiscard]] double getProbability(double x) const override;

    /**
     * @brief Log-PDF: −log(2b) − |x−μ|/b.
     */
    [[nodiscard]] double getLogProbability(double x) const override;

    /**
     * @brief CDF: piecewise via signed exponential decay from μ.
     * F(x) = 0.5·exp((x−μ)/b) for x ≤ μ; 1 − 0.5·exp(−(x−μ)/b) for x > μ.
     */
    [[nodiscard]] double getCumulativeProbability(double x) const override;

    /**
     * @brief Quantile (inverse CDF): μ + b·sign(p−0.5)·log(1−2|p−0.5|).
     * Closed form — no iterative solver required.
     * @throws std::invalid_argument if p not in [0, 1]
     */
    [[nodiscard]] double getQuantile(double p) const override;

    /** @brief Single random sample via inverse CDF. */
    [[nodiscard]] double sample(std::mt19937& rng) const override;

    /** @brief Generate n random samples. */
    [[nodiscard]] std::vector<double> sample(std::mt19937& rng, size_t n) const override;

    //==========================================================================
    // 6. DISTRIBUTION MANAGEMENT
    //==========================================================================

    /**
     * @brief Fit μ and b by MLE.
     *
     * Closed-form estimators:
     *   μ̂ = median(data)
     *   b̂ = (1/n)·Σ|xᵢ − μ̂|
     *
     * Requires O(n log n) sort for the median; O(n) for b̂.
     *
     * @param values Observed data (must be finite, non-empty)
     * @throws std::invalid_argument if values is empty or contains non-finite values
     */
    void fit(const std::vector<double>& values) override;

    /** @brief Parallel batch fitting across multiple independent datasets. */
    static void parallelBatchFit(const std::vector<std::vector<double>>& datasets,
                                 std::vector<LaplaceDistribution>& results);

    /** @brief Reset to default (μ = 0, b = 1). */
    void reset() noexcept override;

    /** @brief String representation. */
    std::string toString() const override;

    //==========================================================================
    // 12. DISTRIBUTION-SPECIFIC UTILITY METHODS
    //==========================================================================

    /** @brief Mode = μ (PMF maximum; Laplace has a sharp peak at the location). */
    [[nodiscard]] double getMode() const noexcept {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        return mu_;
    }

    /** @brief Median = μ (exact; Laplace CDF(μ) = 0.5 by symmetry). */
    [[nodiscard]] double getMedian() const override {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        return mu_;
    }

    /**
     * @brief Entropy = 1 + log(2b)  (nats).
     * Independent of μ; increasing in b.
     */
    [[nodiscard]] double getEntropy() const override;

    /** @brief Check if this is the standard Laplace (μ=0, b=1). */
    [[nodiscard]] bool isStandard() const noexcept {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        return std::fabs(mu_) < detail::DEFAULT_TOLERANCE &&
               std::fabs(b_ - detail::ONE) < detail::DEFAULT_TOLERANCE;
    }

    //==========================================================================
    // 13. SMART AUTO-DISPATCH BATCH OPERATIONS
    //==========================================================================

    /**
     * @brief Batch PDF via SIMD-accelerated pipeline (see class-level doc for steps).
     * Uses vector_exp after scalar_add(−μ) → fabs loop → scalar_multiply(−1/b).
     */
    void getProbability(std::span<const double> values, std::span<double> results,
                        const detail::PerformanceHint& hint = {}) const;

    /**
     * @brief Batch log-PDF via four-step pipeline; CDF path auto-dispatches.
     */
    void getLogProbability(std::span<const double> values, std::span<double> results,
                           const detail::PerformanceHint& hint = {}) const;

    /**
     * @brief Batch CDF via scalar piecewise loop; parallel for large batches.
     */
    void getCumulativeProbability(std::span<const double> values, std::span<double> results,
                                  const detail::PerformanceHint& hint = {}) const;

    //==========================================================================
    // 15. COMPARISON OPERATORS
    //==========================================================================

    bool operator==(const LaplaceDistribution& other) const;
    bool operator!=(const LaplaceDistribution& other) const;

    //==========================================================================
    // 16. FRIEND FUNCTION STREAM OPERATORS
    //==========================================================================

    friend std::istream& operator>>(std::istream& is, stats::LaplaceDistribution&);
    friend std::ostream& operator<<(std::ostream& os, const stats::LaplaceDistribution&);

   private:
    //==========================================================================
    // 17. PRIVATE FACTORY METHODS
    //==========================================================================

    static LaplaceDistribution createUnchecked(double mu, double b) noexcept;
    LaplaceDistribution(double mu, double b, bool /*bypassValidation*/) noexcept;

    //==========================================================================
    // 18. PRIVATE BATCH IMPLEMENTATION METHODS
    //==========================================================================

    /**
     * @brief SIMD LogPDF/PDF pipeline.
     *
     * LogPDF (4 steps):
     *   Step 1: tmp  = x − μ           [scalar_add(values, −mu_, tmp)]
     *   Step 2: tmp  = |x − μ|         [scalar fabs loop; auto-vectorisable]
     *   Step 3: tmp  = −|x−μ|/b        [scalar_multiply(tmp, neg_inv_b_, tmp)]
     *   Step 4: res  = tmp + neg_log2b_ [scalar_add(tmp, neg_log2b_, results)]
     *
     * PDF: append vector_exp (Step 5).
     */
    void getLogProbabilityBatchUnsafeImpl(const double* values, double* results,
                                          std::size_t count,
                                          double cached_mu, double cached_neg_inv_b,
                                          double cached_neg_log2b) const noexcept;

    void getProbabilityBatchUnsafeImpl(const double* values, double* results,
                                       std::size_t count,
                                       double cached_mu, double cached_neg_inv_b,
                                       double cached_neg_log2b) const noexcept;

    void getCumulativeProbabilityBatchUnsafeImpl(const double* values, double* results,
                                                 std::size_t count,
                                                 double cached_mu,
                                                 double cached_half_inv_b) const noexcept;

    //==========================================================================
    // 19. PRIVATE COMPUTATIONAL METHODS
    //==========================================================================

    void updateCacheUnsafe() const noexcept override;

    static void validateParameters(double mu, double b) {
        if (!std::isfinite(mu))
            throw std::invalid_argument("Location parameter mu must be a finite number");
        if (!std::isfinite(b) || b <= detail::ZERO_DOUBLE)
            throw std::invalid_argument("Scale parameter b must be a positive finite number");
    }

    //==========================================================================
    // 20. DISTRIBUTION PARAMETERS
    //==========================================================================

    /** @brief Location parameter μ. */
    double mu_{detail::ZERO_DOUBLE};

    /** @brief Scale parameter b — must be positive. */
    double b_{detail::ONE};

    /** @brief Atomic copies for lock-free parameter access. */
    mutable std::atomic<double> atomicMu_{detail::ZERO_DOUBLE};
    mutable std::atomic<double> atomicB_{detail::ONE};
    mutable std::atomic<bool>   atomicParamsValid_{false};

    //==========================================================================
    // 21. PERFORMANCE CACHE
    //==========================================================================

    /** @brief −1/b — multiplied by |x−μ| in the LogPDF pipeline. */
    mutable double neg_inv_b_{-detail::ONE};

    /** @brief −log(2b) — constant term of LogPDF. */
    mutable double neg_log2b_{-detail::LN2};

    /** @brief 0.5/b = 0.5 * (−neg_inv_b_) — used in CDF pipeline. */
    mutable double half_inv_b_{detail::HALF};
};

}  // namespace stats

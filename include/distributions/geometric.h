#pragma once

#include "libstats/common/distribution_common.h"

// Geometric is a thin delegation wrapper over NegativeBinomialDistribution(r=1, p)
#include "negative_binomial.h"

namespace stats {

/**
 * @brief Thread-safe Geometric Distribution.
 *
 * @details Models the number of failures k before the first success in a
 * sequence of independent Bernoulli trials, each with success probability p.
 * Convention: X ∈ {0, 1, 2, …} (failures before first success), matching the
 * NegativeBinomial(r=1, p) delegation.
 *
 * @par Mathematical Definition:
 * - PMF:    P(X = k) = p·(1−p)^k  for k ∈ {0, 1, 2, …}
 * - LogPMF: log(p) + k·log(1−p)
 * - CDF:    1 − (1−p)^(k+1)
 * - Quantile: ⌈log(1−q)/log(1−p)⌉ − 1  (closed form)
 * - Parameters: p ∈ (0, 1] (success probability)
 * - Support: k ∈ {0, 1, 2, …}
 *
 * @par Moments:
 * - Mean:     (1−p)/p
 * - Variance: (1−p)/p²
 * - Mode:     0  (PMF is strictly decreasing for any valid p)
 * - Median:   ⌈−ln 2 / ln(1−p)⌉ − 1
 * - Skewness: (2−p)/√(1−p)
 * - Excess kurtosis: 6 + p²/(1−p)
 * - Entropy:  [−(1−p)·ln(1−p) − p·ln p] / p  (nats)
 *
 * @par Delegation Design Pattern:
 * GeometricDistribution holds a private `NegativeBinomialDistribution negbinom_`
 * always kept in sync as NegBinomial(r=1, p). All probability, log-probability,
 * CDF, batch, quantile, and sampling operations are one-line pass-throughs to
 * `negbinom_`. This automatically inherits any future SIMD or parallel improvements
 * to NegativeBinomialDistribution.
 *
 * @par MLE:
 * p̂ = 1 / (1 + x̄)  (closed form from sample mean x̄ of failure counts)
 *
 * @par Thread Safety:
 * All methods are fully thread-safe. GeometricDistribution's `cache_mutex_` and
 * `negbinom_`'s own mutex are independent; the two-phase setter pattern (update
 * `p_` under our lock, then call `negbinom_.trySetP()` outside it) avoids
 * holding two mutexes simultaneously.
 *
 * @author libstats Development Team
 * @version 2.0.0
 * @since 2.0.0
 */
class GeometricDistribution : public DistributionBase {
   public:
    // Dispatch metadata
    static constexpr detail::DistributionType kDistributionType =
        detail::DistributionType::GEOMETRIC;
    static constexpr bool kIsDiscrete = true;

   public:
    //==========================================================================
    // 1. CONSTRUCTORS AND DESTRUCTOR
    //==========================================================================

    /**
     * @brief Construct a Geometric distribution.
     * @param p Success probability (p ∈ (0, 1], default 0.5)
     * @throws std::invalid_argument if p is not in (0, 1]
     */
    explicit GeometricDistribution(double p = detail::HALF);

    /** @brief Thread-safe copy constructor. */
    GeometricDistribution(const GeometricDistribution& other);

    /** @brief Copy assignment operator. */
    GeometricDistribution& operator=(const GeometricDistribution& other);

    /** @brief Move constructor. */
    GeometricDistribution(GeometricDistribution&& other) noexcept;

    /** @brief Move assignment operator. */
    GeometricDistribution& operator=(GeometricDistribution&& other) noexcept;

    /** @brief Destructor — defaulted. */
    ~GeometricDistribution() override = default;

    //==========================================================================
    // 2. SAFE FACTORY METHODS (Exception-free construction)
    //==========================================================================

    /**
     * @brief Safely create a Geometric distribution without throwing exceptions.
     * @param p Success probability (must be in (0, 1])
     * @return Result containing either a valid GeometricDistribution or error info
     */
    [[nodiscard]] static Result<GeometricDistribution> create(double p = detail::HALF) {
        auto v = validateGeometricParameters(p);
        if (v.isError())
            return Result<GeometricDistribution>::makeError(v.error_code, v.message);
        return Result<GeometricDistribution>::ok(createUnchecked(p));
    }

    //==========================================================================
    // 3. PARAMETER GETTERS AND SETTERS
    //==========================================================================

    /** @brief Get success probability p. */
    [[nodiscard]] double getP() const noexcept {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        return p_;
    }

    /** @brief Lock-free atomic getter for p. */
    [[nodiscard]] double getPAtomic() const noexcept;

    /**
     * @brief Set success probability p.
     * @param p New success probability (must be in (0, 1])
     * @throws std::invalid_argument if p is invalid
     */
    void setP(double p);

    /** @brief Alias for setP (single-parameter distributions use setParameters). */
    void setParameters(double p) { setP(p); }

    /** @brief Mean = (1−p)/p. */
    [[nodiscard]] double getMean() const override {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        return (detail::ONE - p_) / p_;
    }

    /** @brief Variance = (1−p)/p². */
    [[nodiscard]] double getVariance() const override {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        return (detail::ONE - p_) / (p_ * p_);
    }

    /** @brief Skewness = (2−p)/√(1−p). */
    [[nodiscard]] double getSkewness() const override {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        return (detail::TWO - p_) / std::sqrt(detail::ONE - p_);
    }

    /** @brief Excess kurtosis = 6 + p²/(1−p). */
    [[nodiscard]] double getKurtosis() const override {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        return 6.0 + (p_ * p_) / (detail::ONE - p_);
    }

    /** @brief Number of parameters (always 1). */
    [[nodiscard]] int getNumParameters() const noexcept override { return 1; }

    /** @brief Distribution name. */
    [[nodiscard]] std::string_view getDistributionName() const noexcept override {
        return "GeometricDistribution";
    }

    /** @brief Geometric is discrete. */
    [[nodiscard]] bool isDiscrete() const noexcept override { return true; }

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

    [[nodiscard]] VoidResult trySetP(double p) noexcept;
    [[nodiscard]] VoidResult trySetParameters(double p) noexcept { return trySetP(p); }
    [[nodiscard]] VoidResult validateCurrentParameters() const noexcept;

    //==========================================================================
    // 5. CORE PROBABILITY METHODS — delegated to negbinom_
    //==========================================================================

    /** @brief PMF at k — delegates to NegativeBinomialDistribution(r=1). */
    [[nodiscard]] double getProbability(double x) const override {
        return negbinom_.getProbability(x);
    }

    /** @brief Log-PMF at k — delegates to NegativeBinomialDistribution(r=1). */
    [[nodiscard]] double getLogProbability(double x) const override {
        return negbinom_.getLogProbability(x);
    }

    /** @brief CDF at k — delegates to NegativeBinomialDistribution(r=1). */
    [[nodiscard]] double getCumulativeProbability(double x) const override {
        return negbinom_.getCumulativeProbability(x);
    }

    /**
     * @brief Quantile (smallest k ≥ 0 s.t. CDF(k) ≥ prob) — delegates.
     * @throws std::invalid_argument if prob not in [0, 1]
     */
    [[nodiscard]] double getQuantile(double prob) const override {
        return negbinom_.getQuantile(prob);
    }

    /** @brief Generate a single random sample — delegates. */
    [[nodiscard]] double sample(std::mt19937& rng) const override {
        return negbinom_.sample(rng);
    }

    /** @brief Generate n random samples — delegates. */
    [[nodiscard]] std::vector<double> sample(std::mt19937& rng, size_t n) const override {
        return negbinom_.sample(rng, n);
    }

    //==========================================================================
    // 6. DISTRIBUTION MANAGEMENT
    //==========================================================================

    /**
     * @brief Fit p by MLE from observed failure counts.
     *
     * MLE for Geometric(p): p̂ = 1 / (1 + x̄) where x̄ is the sample mean.
     * Closed form — no iterative solver required.
     *
     * @param values Non-negative integer failure counts
     * @throws std::invalid_argument if values is empty or contains negative values
     */
    void fit(const std::vector<double>& values) override;

    /** @brief Parallel batch fitting across multiple datasets. */
    static void parallelBatchFit(const std::vector<std::vector<double>>& datasets,
                                 std::vector<GeometricDistribution>& results);

    /** @brief Reset to default (p = 0.5). */
    void reset() noexcept override;

    /** @brief String representation. */
    std::string toString() const override;

    //==========================================================================
    // 12. DISTRIBUTION-SPECIFIC UTILITY METHODS
    //==========================================================================

    /**
     * @brief Mode = 0 (PMF is strictly decreasing for any p ∈ (0,1]).
     */
    [[nodiscard]] double getMode() const noexcept { return detail::ZERO_DOUBLE; }

    /**
     * @brief Median = ⌈−ln 2 / ln(1−p)⌉ − 1.
     *
     * Special case: p = 1 → all probability at k=0 → median = 0.
     */
    [[nodiscard]] double getMedian() const override;

    /**
     * @brief Entropy = [−(1−p)·ln(1−p) − p·ln p] / p  (nats).
     */
    [[nodiscard]] double getEntropy() const override;

    //==========================================================================
    // 13. SMART AUTO-DISPATCH BATCH OPERATIONS — delegated to negbinom_
    //==========================================================================

    /** @brief Batch PMF — SIMD/parallel dispatch via NegativeBinomialDistribution. */
    void getProbability(std::span<const double> values, std::span<double> results,
                        const detail::PerformanceHint& hint = {}) const {
        negbinom_.getProbability(values, results, hint);
    }

    /** @brief Batch log-PMF — SIMD/parallel dispatch via NegativeBinomialDistribution. */
    void getLogProbability(std::span<const double> values, std::span<double> results,
                           const detail::PerformanceHint& hint = {}) const {
        negbinom_.getLogProbability(values, results, hint);
    }

    /** @brief Batch CDF — SIMD/parallel dispatch via NegativeBinomialDistribution. */
    void getCumulativeProbability(std::span<const double> values, std::span<double> results,
                                  const detail::PerformanceHint& hint = {}) const {
        negbinom_.getCumulativeProbability(values, results, hint);
    }

    //==========================================================================
    // 15. COMPARISON OPERATORS
    //==========================================================================

    bool operator==(const GeometricDistribution& other) const;
    bool operator!=(const GeometricDistribution& other) const;

    //==========================================================================
    // 16. FRIEND FUNCTION STREAM OPERATORS
    //==========================================================================

    friend std::istream& operator>>(std::istream& is, stats::GeometricDistribution&);
    friend std::ostream& operator<<(std::ostream& os, const stats::GeometricDistribution&);

   private:
    //==========================================================================
    // 17. PRIVATE FACTORY METHODS
    //==========================================================================

    static GeometricDistribution createUnchecked(double p) noexcept;
    GeometricDistribution(double p, bool /*bypassValidation*/) noexcept;

    //==========================================================================
    // 18. PRIVATE BATCH IMPLEMENTATION METHODS
    //==========================================================================

    // Not needed: all batch operations delegate to negbinom_.
    // Section retained for template compliance.

    //==========================================================================
    // 19. PRIVATE COMPUTATIONAL METHODS
    //==========================================================================

    // Not needed: all computation delegated to negbinom_ or computed inline.
    // Section retained for template compliance.

    //==========================================================================
    // 20. PRIVATE CACHE MANAGEMENT
    //==========================================================================

    /**
     * @brief Sync negbinom_ with current p_ and mark cache valid.
     *
     * Calls negbinom_.trySetP(p_) to update the delegate's internals, then
     * sets cache_valid_ = true on this object. Called within a held
     * unique_lock on cache_mutex_; negbinom_ acquires its own mutex internally
     * — no lock-ordering conflict since negbinom_ is private.
     */
    void updateCacheUnsafe() const noexcept override;

    static void validateParameters(double p) {
        if (!std::isfinite(p) || p <= detail::ZERO_DOUBLE || p > detail::ONE)
            throw std::invalid_argument("Success probability p must be in (0, 1]");
    }

    //==========================================================================
    // 21. DISTRIBUTION PARAMETERS
    //==========================================================================

    /** @brief Success probability p — must be in (0, 1]. API copy for O(1) reads. */
    double p_{detail::HALF};

    /** @brief Atomic copy for lock-free parameter access. */
    mutable std::atomic<double> atomicP_{detail::HALF};
    mutable std::atomic<bool>   atomicParamsValid_{false};

    //==========================================================================
    // 22. DELEGATE DISTRIBUTION
    //==========================================================================

    /**
     * @brief Delegate: NegativeBinomial(r=1, p) — identical to Geometric(p).
     *
     * Invariant: negbinom_.getR() == 1.0 and negbinom_.getP() == p_ at all times.
     * All probability, batch, quantile, and sampling calls pass through here.
     */
    mutable NegativeBinomialDistribution negbinom_{detail::ONE, detail::HALF};
};

}  // namespace stats

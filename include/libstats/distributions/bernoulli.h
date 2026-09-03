#pragma once

#include "libstats/common/distribution_common.h"

// Bernoulli is a thin delegation wrapper over BinomialDistribution(n=1, p)
#include "binomial.h"

namespace stats {

/**
 * @brief Thread-safe Bernoulli Distribution.
 *
 * @details Models a single trial with two outcomes: success (X=1) with
 * probability p, failure (X=0) with probability 1-p. This is the n=1 special
 * case of the Binomial distribution.
 *
 * @par Mathematical Definition:
 * - PMF:    P(X=1) = p, P(X=0) = 1-p
 * - LogPMF: log(p) at x=1, log(1-p) at x=0
 * - CDF:    0 for x<0, 1-p for 0<=x<1, 1 for x>=1
 * - Quantile: right-continuous discrete inverse Q(q) = min{k : F(k) >= q}
 *             (#104 contract): Q(q) = 0 for q <= 1-p, else 1; never NaN for
 *             q in (0,1).
 * - Parameters: p in [0, 1] (success probability)
 * - Support: x in {0, 1}
 *
 * @par Moments:
 * - Mean:     p
 * - Variance: p*(1-p)
 * - Mode:     1 if p>0.5, 0 if p<0.5, 0 by convention at the p=0.5 tie
 * - Median:   1 if p>0.5, 0 if p<0.5, 0.5 by convention at the p=0.5 tie
 * - Skewness: (1-2p)/sqrt(p*(1-p))
 * - Excess kurtosis: (1-6p*(1-p))/(p*(1-p))
 * - Entropy:  -p*log(p) - (1-p)*log(1-p)  (nats; 0*log(0) := 0)
 *
 * @par Delegation Design Pattern:
 * BernoulliDistribution holds a private `BinomialDistribution binomial_`
 * always kept in sync as Binomial(n=1, p). All probability, log-probability,
 * CDF, quantile, batch, sampling, and moment operations are one-line
 * pass-throughs to `binomial_` -- Bernoulli(p) is Binomial(1,p) *exactly*,
 * with no algebraic simplification available (unlike e.g. Geometric, whose
 * own moment formulas are simpler than the general NegativeBinomial ones),
 * so full delegation (matching ChiSquared's style) is simplest here.
 * BinomialDistribution::getProbability/getLogProbability/getCumulativeProbability
 * all guard `!std::isfinite(x)` at the top (see binomial.cpp), so the ±inf/NaN
 * contract (#103) holds for Bernoulli with no extra work, scalar and batch.
 *
 * @par Parameter validation — deliberately [0, 1], not (0, 1]:
 * This follows BinomialDistribution's own convention exactly (see
 * `validateBernoulliParameters` in core/error_handling.h): both p=0 (point
 * mass at 0) and p=1 (point mass at 1) are valid degenerate distributions,
 * matching Binomial(n=1, p=0) and Binomial(n=1, p=1). This differs from
 * GeometricDistribution's (0, 1] convention, which excludes p=0 because
 * Geometric's infinite support would not sum to 1 there.
 *
 * @par MLE:
 * p_hat = x_bar (sample mean of 0/1 data) — closed form.
 *
 * @par Thread Safety:
 * All methods are fully thread-safe. BernoulliDistribution's `cache_mutex_`
 * and `binomial_`'s own mutex are independent; the two-phase setter pattern
 * (update `p_` under our lock, then call `binomial_.trySetP()` outside it)
 * avoids holding two mutexes simultaneously.
 *
 * @author libstats Development Team
 * @version 2.4.0
 * @since 2.4.0
 */
class BernoulliDistribution : public DistributionBase {
   public:
    // Dispatch metadata
    static constexpr detail::DistributionType kDistributionType =
        detail::DistributionType::BERNOULLI;
    static constexpr bool kIsDiscrete = true;

   public:
    //==========================================================================
    // 1. CONSTRUCTORS AND DESTRUCTOR
    //==========================================================================

    /**
     * @brief Construct a Bernoulli distribution.
     * @param p Success probability (p in [0, 1], default 0.5)
     * @throws std::invalid_argument if p is not in [0, 1]
     */
    explicit BernoulliDistribution(double p = detail::HALF);

    /** @brief Thread-safe copy constructor. */
    BernoulliDistribution(const BernoulliDistribution& other);

    /** @brief Copy assignment operator. */
    BernoulliDistribution& operator=(const BernoulliDistribution& other);

    /** @brief Move constructor. */
    BernoulliDistribution(BernoulliDistribution&& other) noexcept;

    /** @brief Move assignment operator. */
    BernoulliDistribution& operator=(BernoulliDistribution&& other) noexcept;

    /** @brief Destructor — defaulted. */
    ~BernoulliDistribution() override = default;

    //==========================================================================
    // 2. SAFE FACTORY METHODS (Exception-free construction)
    //==========================================================================

    /**
     * @brief Safely create a Bernoulli distribution without throwing exceptions.
     * @param p Success probability (must be in [0, 1])
     * @return Result containing either a valid BernoulliDistribution or error info
     */
    [[nodiscard]] static Result<BernoulliDistribution> create(double p = detail::HALF) {
        auto v = validateBernoulliParameters(p);
        if (v.isError())
            return Result<BernoulliDistribution>::makeError(v.errorCode(), v.message());
        return Result<BernoulliDistribution>::ok(createUnchecked(p));
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
     * @param p New success probability (must be in [0, 1])
     * @throws std::invalid_argument if p is invalid
     */
    void setP(double p);

    /** @brief Alias for setP (single-parameter distributions use setParameters). */
    void setParameters(double p) { setP(p); }

    /** @brief Mean = p — delegates to BinomialDistribution(n=1). */
    [[nodiscard]] double getMean() const override { return binomial_.getMean(); }

    /** @brief Variance = p*(1-p) — delegates to BinomialDistribution(n=1). */
    [[nodiscard]] double getVariance() const override { return binomial_.getVariance(); }

    /** @brief Skewness = (1-2p)/sqrt(p*(1-p)) — delegates. */
    [[nodiscard]] double getSkewness() const override { return binomial_.getSkewness(); }

    /** @brief Excess kurtosis = (1-6p(1-p))/(p*(1-p)) — delegates. */
    [[nodiscard]] double getKurtosis() const override { return binomial_.getKurtosis(); }

    /** @brief Number of parameters (always 1). */
    [[nodiscard]] int getNumParameters() const noexcept override { return 1; }

    /** @brief Distribution name. */
    [[nodiscard]] std::string_view getDistributionName() const noexcept override {
        return "Bernoulli";
    }

    /** @brief Bernoulli is discrete. */
    [[nodiscard]] bool isDiscrete() const noexcept override { return true; }

    /** @brief Support lower bound: 0. */
    [[nodiscard]] double getSupportLowerBound() const noexcept override {
        return detail::ZERO_DOUBLE;
    }

    /** @brief Support upper bound: 1. */
    [[nodiscard]] double getSupportUpperBound() const noexcept override { return detail::ONE; }

    //==========================================================================
    // 4. RESULT-BASED SETTERS
    //==========================================================================

    [[nodiscard]] VoidResult trySetP(double p) noexcept;
    [[nodiscard]] VoidResult trySetParameters(double p) noexcept { return trySetP(p); }
    [[nodiscard]] VoidResult validateCurrentParameters() const noexcept;

    //==========================================================================
    // 5. CORE PROBABILITY METHODS — delegated to binomial_
    //==========================================================================

    /** @brief PMF at x — delegates to BinomialDistribution(n=1). */
    [[nodiscard]] double getProbability(double x) const override {
        return binomial_.getProbability(x);
    }

    /** @brief Log-PMF at x — delegates to BinomialDistribution(n=1). */
    [[nodiscard]] double getLogProbability(double x) const override {
        return binomial_.getLogProbability(x);
    }

    /** @brief CDF at x — delegates to BinomialDistribution(n=1). */
    [[nodiscard]] double getCumulativeProbability(double x) const override {
        return binomial_.getCumulativeProbability(x);
    }

    /**
     * @brief Quantile: right-continuous discrete inverse Q(q) = min{k : F(k) >= q}
     * (#104 contract) — delegates to BinomialDistribution(n=1), whose own
     * getQuantile already implements exactly this search over {0,...,n}.
     * @throws std::invalid_argument if q not in [0, 1]
     */
    [[nodiscard]] double getQuantile(double q) const override { return binomial_.getQuantile(q); }

    /** @brief Generate a single random sample — delegates. */
    [[nodiscard]] double sample(std::mt19937& rng) const override { return binomial_.sample(rng); }

    /** @brief Generate n random samples — delegates. */
    [[nodiscard]] std::vector<double> sample(std::mt19937& rng, size_t n) const override {
        return binomial_.sample(rng, n);
    }

    //==========================================================================
    // 6. DISTRIBUTION MANAGEMENT
    //==========================================================================

    /**
     * @brief Fit p by MLE from observed 0/1 outcomes.
     *
     * MLE for Bernoulli(p): p_hat = x_bar, the sample mean of 0/1 data.
     * Closed form — no iterative solver required.
     *
     * @param values 0/1-valued observations
     * @throws std::invalid_argument if values is empty or contains a value
     *         other than 0.0 or 1.0
     */
    void fit(const std::vector<double>& values) override;

    /** @brief Parallel batch fitting across multiple datasets. */
    static void parallelBatchFit(const std::vector<std::vector<double>>& datasets,
                                 std::vector<BernoulliDistribution>& results);

    /** @brief Reset to default (p = 0.5). */
    void reset() noexcept override;

    /** @brief String representation. */
    std::string toString() const override;

    //==========================================================================
    // 12. DISTRIBUTION-SPECIFIC UTILITY METHODS
    //==========================================================================

    /**
     * @brief Mode: 1 if p>0.5, 0 if p<0.5, 0 by convention when p=0.5 (tie).
     */
    [[nodiscard]] double getMode() const noexcept;

    /**
     * @brief Median: 1 if p>0.5, 0 if p<0.5, 0.5 by convention when p=0.5.
     */
    [[nodiscard]] double getMedian() const override;

    /**
     * @brief Entropy = -p*log(p) - (1-p)*log(1-p) (nats) — delegates to
     * BinomialDistribution(n=1), whose own exact-PMF-summation entropy
     * reduces to exactly this formula at n=1.
     */
    [[nodiscard]] double getEntropy() const override { return binomial_.getEntropy(); }

    //==========================================================================
    // 13. SMART AUTO-DISPATCH BATCH OPERATIONS — delegated to binomial_
    //==========================================================================
    // For all three overloads below: values and results must have the same
    // size (a mismatch throws std::invalid_argument) and must not overlap (#112).
    // An in-place call silently returns wrong values; overlap is caught only by
    // a debug-mode assert in detail::DispatchUtils. Full contract in
    // core/distribution_interface.h.

    /** @brief Batch PMF — SIMD/parallel dispatch via BinomialDistribution. */
    void getProbability(std::span<const double> values, std::span<double> results,
                        const detail::PerformanceHint& hint = {}) const {
        binomial_.getProbability(values, results, hint);
    }

    /** @brief Batch log-PMF — SIMD/parallel dispatch via BinomialDistribution. */
    void getLogProbability(std::span<const double> values, std::span<double> results,
                           const detail::PerformanceHint& hint = {}) const {
        binomial_.getLogProbability(values, results, hint);
    }

    /** @brief Batch CDF — SIMD/parallel dispatch via BinomialDistribution. */
    void getCumulativeProbability(std::span<const double> values, std::span<double> results,
                                  const detail::PerformanceHint& hint = {}) const {
        binomial_.getCumulativeProbability(values, results, hint);
    }

    //==========================================================================
    // 15. COMPARISON OPERATORS
    //==========================================================================

    bool operator==(const BernoulliDistribution& other) const;
    bool operator!=(const BernoulliDistribution& other) const;

    //==========================================================================
    // 16. FRIEND FUNCTION STREAM OPERATORS
    //==========================================================================

    friend std::istream& operator>>(std::istream& is, stats::BernoulliDistribution&);
    friend std::ostream& operator<<(std::ostream& os, const stats::BernoulliDistribution&);

   private:
    //==========================================================================
    // 17. PRIVATE FACTORY METHODS
    //==========================================================================

    static BernoulliDistribution createUnchecked(double p) noexcept;
    BernoulliDistribution(double p, bool /*bypassValidation*/) noexcept;

    //==========================================================================
    // 18. PRIVATE BATCH IMPLEMENTATION METHODS
    //==========================================================================

    // Not needed: all batch operations delegate to binomial_.
    // Section retained for template compliance.

    //==========================================================================
    // 19. PRIVATE COMPUTATIONAL METHODS
    //==========================================================================

    // Not needed: all computation delegated to binomial_ or computed inline.
    // Section retained for template compliance.

    //==========================================================================
    // 20. PRIVATE CACHE MANAGEMENT
    //==========================================================================

    /**
     * @brief Sync binomial_ with current p_ and mark cache valid.
     *
     * Calls binomial_.trySetP(p_) to update the delegate's internals (n stays
     * fixed at 1), then sets cache_valid_ = true on this object. Called
     * within a held unique_lock on cache_mutex_; binomial_ acquires its own
     * mutex internally — no lock-ordering conflict since binomial_ is
     * private.
     */
    void updateCacheUnsafe() const noexcept override;

    static void validateParameters(double p) {
        if (std::isnan(p) || std::isinf(p) || p < detail::ZERO_DOUBLE || p > detail::ONE)
            throw std::invalid_argument("Success probability p must be in [0, 1]");
    }

    //==========================================================================
    // 21. DISTRIBUTION PARAMETERS
    //==========================================================================

    /** @brief Success probability p — must be in [0, 1]. API copy for O(1) reads. */
    double p_{detail::HALF};

    /** @brief Atomic copy for lock-free parameter access. */
    mutable std::atomic<double> atomicP_{detail::HALF};
    mutable std::atomic<bool> atomicParamsValid_{false};

    //==========================================================================
    // 22. DELEGATE DISTRIBUTION
    //==========================================================================

    /**
     * @brief Delegate: Binomial(n=1, p) — identical to Bernoulli(p).
     *
     * Invariant: binomial_.getN() == 1 and binomial_.getP() == p_ at all times.
     * All probability, moment, batch, quantile, and sampling calls pass
     * through here.
     */
    mutable BinomialDistribution binomial_{1, detail::HALF};
};

}  // namespace stats

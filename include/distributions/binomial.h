#pragma once

#include "libstats/common/distribution_common.h"
#include "libstats/common/distribution_platform_common.h"

namespace stats {

/**
 * @brief Thread-safe Binomial Distribution for modelling success counts.
 *
 * @details Models the number of successes k in n independent Bernoulli trials
 * each with success probability p.
 *
 * @par Mathematical Definition:
 * - PMF:    P(X = k) = C(n,k)·p^k·(1−p)^(n−k)  for k ∈ {0,…,n}
 * - LogPMF: lgamma(n+1) − lgamma(k+1) − lgamma(n−k+1) + k·log(p) + (n−k)·log(1−p)
 * - CDF:    I_{1−p}(n−k, k+1)  via the regularized incomplete beta (existing detail::beta_i)
 * - Quantile: smallest k ∈ {0,…,n} such that CDF(k) ≥ p
 * - Parameters: n > 0 (integer trials), p ∈ [0,1] (success probability)
 * - Support: k ∈ {0, 1, …, n}
 *
 * @par Moments:
 * - Mean:     n·p
 * - Variance: n·p·(1−p)
 * - Mode:     floor((n+1)·p) or floor((n+1)·p)−1 when (n+1)·p is integer
 * - Skewness: (1−2p)/√(n·p·(1−p))
 * - Excess kurtosis: (1−6p(1−p))/(n·p·(1−p))
 *
 * @par Batch operations:
 * The VECTORIZED path uses a scalar loop — lgamma(k+1) and lgamma(n−k+1) are
 * not in VectorOps.  The value over per-element calls is: logNFact_ = lgamma(n+1),
 * logP_, and log1mP_ are cached loop-invariants, avoiding repeated log and lgamma
 * computations.  The PARALLEL strategy provides genuine multi-core throughput.
 *
 * @par MLE:
 * Given fixed n: p̂ = k̄/n (one-pass, closed-form).
 * When n is unknown: n is estimated as max(observed k) — the smallest n consistent
 * with all observations (following libhmm's EM-compatible approach).
 *
 * @author libstats Development Team
 * @version 1.3.0
 * @since 1.3.0
 */
class BinomialDistribution : public DistributionBase {
   public:
    // Dispatch metadata — replaces DistributionTraits<BinomialDistribution> (v2.0.0)
    static constexpr detail::DistributionType kDistributionType =
        detail::DistributionType::BINOMIAL;
    static constexpr bool kIsDiscrete = true;

   public:
    //==========================================================================
    // 1. CONSTRUCTORS AND DESTRUCTOR
    //==========================================================================

    /**
     * @brief Construct a Binomial distribution.
     * @param n Number of trials (positive integer, default 10)
     * @param p Success probability (in [0,1], default 0.5)
     * @throws std::invalid_argument if n ≤ 0 or p not in [0,1]
     */
    explicit BinomialDistribution(int n = 10, double p = detail::HALF);

    /** @brief Thread-safe copy constructor. */
    BinomialDistribution(const BinomialDistribution& other);

    /** @brief Copy assignment operator. */
    BinomialDistribution& operator=(const BinomialDistribution& other);

    /** @brief Move constructor. */
    BinomialDistribution(BinomialDistribution&& other) noexcept;

    /** @brief Move assignment operator. @warning NOT noexcept. */
    BinomialDistribution& operator=(BinomialDistribution&& other) noexcept;

    /** @brief Destructor — defaulted. */
    ~BinomialDistribution() override = default;

    //==========================================================================
    // 2. SAFE FACTORY METHODS (Exception-free construction)
    //==========================================================================

    [[nodiscard]] static Result<BinomialDistribution> create(int n = 10,
                                                             double p = detail::HALF) noexcept {
        auto v = validateBinomialParameters(n, p);
        if (v.isError())
            return Result<BinomialDistribution>::makeError(v.error_code, v.message);
        return Result<BinomialDistribution>::ok(createUnchecked(n, p));
    }

    //==========================================================================
    // 3. PARAMETER GETTERS AND SETTERS
    //==========================================================================

    /** @brief Get number of trials n. */
    [[nodiscard]] int getN() const noexcept {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        return n_;
    }

    /** @brief Get success probability p. */
    [[nodiscard]] double getP() const noexcept {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        return p_;
    }

    /** @brief Lock-free atomic getter for n. */
    [[nodiscard]] int getNAtomic() const noexcept;

    /** @brief Lock-free atomic getter for p. */
    [[nodiscard]] double getPAtomic() const noexcept;

    /**
     * @brief Set number of trials n.
     * @throws std::invalid_argument if n ≤ 0
     */
    void setN(int n);

    /**
     * @brief Set success probability p.
     * @throws std::invalid_argument if p not in [0,1]
     */
    void setP(double p);

    /**
     * @brief Set both parameters simultaneously.
     */
    void setParameters(int n, double p);

    /** @brief Mean = n·p. */
    [[nodiscard]] double getMean() const noexcept override;

    /** @brief Variance = n·p·(1−p). */
    [[nodiscard]] double getVariance() const noexcept override;

    /** @brief Skewness = (1−2p)/√(n·p·(1−p)). */
    [[nodiscard]] double getSkewness() const noexcept override;

    /** @brief Excess kurtosis = (1−6p(1−p))/(n·p·(1−p)). */
    [[nodiscard]] double getKurtosis() const noexcept override;

    /** @brief Number of parameters (always 2). */
    [[nodiscard]] int getNumParameters() const noexcept override { return 2; }

    /** @brief Distribution name. */
    [[nodiscard]] std::string getDistributionName() const override {
        return "BinomialDistribution";
    }

    /** @brief Binomial is discrete. */
    [[nodiscard]] bool isDiscrete() const noexcept override { return true; }

    /** @brief Support lower bound: 0. */
    [[nodiscard]] double getSupportLowerBound() const noexcept override {
        return detail::ZERO_DOUBLE;
    }

    /** @brief Support upper bound: n. */
    [[nodiscard]] double getSupportUpperBound() const noexcept override {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        return static_cast<double>(n_);
    }

    //==========================================================================
    // 4. RESULT-BASED SETTERS
    //==========================================================================

    [[nodiscard]] VoidResult trySetN(int n) noexcept;
    [[nodiscard]] VoidResult trySetP(double p) noexcept;
    [[nodiscard]] VoidResult trySetParameters(int n, double p) noexcept;
    [[nodiscard]] VoidResult validateCurrentParameters() const noexcept;

    //==========================================================================
    // 5. CORE PROBABILITY METHODS
    //==========================================================================

    /**
     * @brief PMF at k (rounded to nearest integer): C(n,k)·p^k·(1−p)^(n−k).
     * Returns 0 for non-integer or out-of-range values.
     */
    [[nodiscard]] double getProbability(double x) const override;

    /**
     * @brief Log-PMF at k.
     * Returns −∞ for out-of-range values.
     */
    [[nodiscard]] double getLogProbability(double x) const noexcept override;

    /**
     * @brief CDF via regularized incomplete beta I_{1−p}(n−k, k+1).
     * O(1) computation using existing detail::beta_i.
     */
    [[nodiscard]] double getCumulativeProbability(double x) const override;

    /**
     * @brief Smallest k ∈ {0,…,n} such that CDF(k) ≥ p.
     * @throws std::invalid_argument if p not in [0,1]
     */
    [[nodiscard]] double getQuantile(double p) const override;

    /** @brief Sample via std::binomial_distribution<int>(n, p). */
    [[nodiscard]] double sample(std::mt19937& rng) const override;

    /** @brief Generate n random samples. */
    [[nodiscard]] std::vector<double> sample(std::mt19937& rng, size_t n) const override;

    //==========================================================================
    // 6. DISTRIBUTION MANAGEMENT
    //==========================================================================

    /**
     * @brief Fit n and p by MLE.
     *
     * n̂ = max(round(xᵢ)) — smallest n consistent with all observations.
     * p̂ = k̄/n̂  (exact MLE for p given n).
     *
     * @param values Non-negative integer observations
     * @throws std::invalid_argument if values is empty
     */
    void fit(const std::vector<double>& values) override;

    /** @brief Parallel batch fitting. */
    static void parallelBatchFit(const std::vector<std::vector<double>>& datasets,
                                 std::vector<BinomialDistribution>& results);

    /** @brief Reset to default (n=10, p=0.5). */
    void reset() noexcept override;

    /** @brief String representation. */
    std::string toString() const override;

    //==========================================================================
    // 12. DISTRIBUTION-SPECIFIC UTILITY METHODS
    //==========================================================================

    /** @brief Mode = floor((n+1)·p) — with tie-breaking for exact integers. */
    [[nodiscard]] double getMode() const noexcept;

    /** @brief Entropy (bits) ≈ ½ log₂(2πe·n·p·(1−p)) for large n. */
    [[nodiscard]] double getEntropy() const noexcept override;

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

    bool operator==(const BinomialDistribution& other) const;
    bool operator!=(const BinomialDistribution& other) const;

    //==========================================================================
    // 16. FRIEND FUNCTION STREAM OPERATORS
    //==========================================================================

    friend std::istream& operator>>(std::istream& is, stats::BinomialDistribution&);
    friend std::ostream& operator<<(std::ostream& os, const stats::BinomialDistribution&);

   private:
    //==========================================================================
    // 17. PRIVATE FACTORY METHODS
    //==========================================================================

    static BinomialDistribution createUnchecked(int n, double p) noexcept;
    BinomialDistribution(int n, double p, bool /*bypassValidation*/) noexcept;

    /** @brief Compute log C(n,k) = logNFact_ − lgamma(k+1) − lgamma(n−k+1). */
    double logBinomCoeff(int k) const noexcept;

    //==========================================================================
    // 18. PRIVATE BATCH IMPLEMENTATION METHODS
    //==========================================================================

    /**
     * @brief Scalar-loop batch PMF / LogPMF.
     *
     * The VECTORIZED path uses a scalar loop because lgamma(k+1) and
     * lgamma(n−k+1) are not in VectorOps.  Cached loop-invariants:
     *   logNFact_ = lgamma(n+1)  — avoids recomputing lgamma(n+1) per element
     *   logP_  = log(p)          — avoids recomputing log per element
     *   log1mP_ = log(1−p)       — avoids recomputing log(1−p) per element
     *
     * When a vector_lgamma primitive is added to VectorOps, the hot path
     * becomes fully SIMD-accelerated (like the Gaussian erf path).
     * Until then, PARALLEL is the recommended strategy for large batches.
     */
    void getLogProbabilityBatchImpl(const double* values, double* results, std::size_t count,
                                    int cached_n, double cached_logNFact, double cached_logP,
                                    double cached_log1mP) const noexcept;

    void getProbabilityBatchImpl(const double* values, double* results, std::size_t count,
                                 int cached_n, double cached_logNFact, double cached_logP,
                                 double cached_log1mP) const noexcept;

    void getCumulativeProbabilityBatchImpl(const double* values, double* results,
                                           std::size_t count) const noexcept;

    //==========================================================================
    // 19. PRIVATE COMPUTATIONAL METHODS
    //==========================================================================

    void updateCacheUnsafe() const noexcept override;

    static void validateParameters(int n, double p) {
        if (n <= 0)
            throw std::invalid_argument("Number of trials n must be a positive integer");
        if (std::isnan(p) || std::isinf(p) || p < detail::ZERO_DOUBLE || p > detail::ONE)
            throw std::invalid_argument("Success probability p must be in [0, 1]");
    }

    //==========================================================================
    // 20–21. DISTRIBUTION PARAMETERS
    //==========================================================================

    /** @brief Number of trials n — must be positive. */
    int n_{10};

    /** @brief Success probability p — must be in [0, 1]. */
    double p_{detail::HALF};

    /** @brief Atomic copies for lock-free access. */
    mutable std::atomic<int> atomicN_{10};
    mutable std::atomic<double> atomicP_{detail::HALF};
    mutable std::atomic<bool> atomicParamsValid_{false};

    //==========================================================================
    // 22. PERFORMANCE CACHE
    //==========================================================================

    /** @brief lgamma(n+1) — the dominant loop-invariant in LogPMF. */
    mutable double logNFact_{detail::ZERO_DOUBLE};

    /** @brief log(p) — avoids per-element log computation. */
    mutable double logP_{detail::ZERO_DOUBLE};

    /** @brief log(1−p) — avoids per-element log computation. */
    mutable double log1mP_{detail::ZERO_DOUBLE};

    //==========================================================================
    // 23. OPTIMIZATION FLAGS
    //==========================================================================

    //==========================================================================
    // 24. SPECIALIZED CACHES
    //==========================================================================

    // Note: Binomial uses standard caching only.
    // Section maintained for template compliance.
};

}  // namespace stats

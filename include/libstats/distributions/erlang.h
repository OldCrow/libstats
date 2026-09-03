#pragma once

// Common distribution includes (consolidates std library and core headers)
#include "libstats/common/distribution_common.h"

// Erlang is a thin delegation wrapper over GammaDistribution(k, lambda).
// libstats' GammaDistribution is (alpha SHAPE, beta RATE) -- so Erlang(k, lambda)
// with lambda a RATE delegates to GammaDistribution(k, lambda) DIRECTLY. (v2.4.0
// kickoff correction on #55: the original issue note assumed a scale-parameterized
// Gamma and would have delegated to GammaDistribution(k, 1.0/lambda) -- wrong for
// this codebase, where GammaDistribution's second parameter is already a rate.)
#include "gamma.h"

namespace stats {

/**
 * @brief Thread-safe Erlang Distribution: a Gamma distribution restricted to
 * a positive-integer shape parameter.
 *
 * @details The Erlang distribution models the waiting time until the k-th
 * event in a Poisson process with rate lambda -- equivalently, the sum of k
 * i.i.d. Exponential(lambda) random variables. It is the special case of the
 * Gamma distribution with an integer shape parameter.
 *
 * @par Mathematical Definition:
 * - PDF: f(x; k, lambda) = lambda^k * x^(k-1) * exp(-lambda*x) / (k-1)!  for x >= 0
 * - CDF: F(x; k, lambda) = P(k, lambda*x)  (regularized incomplete gamma function)
 * - Parameters: k >= 1 (positive integer shape), lambda > 0 (rate)
 * - Support: x in [0, infinity)
 * - Mean: k/lambda
 * - Variance: k/lambda^2
 * - Mode: (k-1)/lambda
 *
 * @par Relationship to Gamma:
 * Erlang(k, lambda) = Gamma(alpha=k, beta=lambda) exactly, using libstats'
 * shape-RATE parameterization for Gamma (see gamma.h).
 *
 * @par Design decision — k as `int`, not `double`:
 * Unlike ChiSquaredDistribution (whose degrees-of-freedom `k_` is a `double`,
 * matching Gamma's `alpha` directly, since any positive real k is valid),
 * Erlang's shape parameter is mathematically restricted to positive integers.
 * This implementation accepts `int k` (mirroring BinomialDistribution's `int
 * n` convention for its own integer-only parameter) rather than `double k`
 * plus a runtime integrality check. This removes any ambiguity about "how
 * close to an integer counts as integral" and any need for a separate
 * rejection path for non-integer doubles -- the type system enforces it.
 *
 * @par Delegation Design Pattern:
 * ErlangDistribution holds a private `GammaDistribution gamma_` always kept
 * in sync as Gamma(k, lambda). Moments, entropy, median, mode, CDF, quantile,
 * and sampling are one-line pass-throughs to `gamma_`.
 *
 * @par ±inf / NaN handling — pure pass-through since #130:
 * PDF/LogPDF are pure delegates too. At first landing they carried their own
 * non-finite guards, because `GammaDistribution::getProbability` /
 * `getLogProbability` then lacked an `isfinite(x)` guard and their log-space
 * formula was NaN at x=+inf for every alpha >= 1 (`0*log(inf)` at alpha=1,
 * `inf - inf` above) — a path *always* live for Erlang since alpha = k >= 1.
 * That defect was fixed at the source in gamma.cpp (#130: scalar guards plus
 * batch fixup handling; pdf(±inf)=0, logpdf(±inf)=-inf, NaN propagates,
 * scalar == batch), so the wrapper-side guards were removed as redundant.
 * The #103/#104 contract tests in tests/test_erlang_enhanced.cpp still pin
 * the behaviour through the delegation.
 *
 * @par MLE:
 * Method of moments: k_hat = round(x_bar^2 / s^2), clamped to >= 1; then
 * lambda_hat = k_hat / x_bar. (Degenerate/near-zero-variance data falls back
 * to k_hat = 1 rather than diverging toward infinity.)
 *
 * @par Thread Safety:
 * All methods are fully thread-safe using reader-writer locks. Like
 * ChiSquaredDistribution, ErlangDistribution has no atomic parameter copies
 * of its own; the atomic fast path is provided entirely by `gamma_`'s own
 * atomics.
 *
 * @author libstats Development Team
 * @version 2.4.0
 * @since 2.4.0
 */
class ErlangDistribution : public DistributionBase {
   public:
    // Dispatch metadata
    static constexpr detail::DistributionType kDistributionType =
        detail::DistributionType::ERLANG;
    static constexpr bool kIsDiscrete = false;

   public:
    //==========================================================================
    // 1. CONSTRUCTORS AND DESTRUCTOR
    //==========================================================================

    /**
     * @brief Constructs an Erlang distribution with the given shape and rate.
     * @param k Shape parameter (positive integer, default: 1)
     * @param lambda Rate parameter (must be positive and finite, default: 1.0)
     * @throws std::invalid_argument if k < 1 or lambda is not positive finite
     */
    explicit ErlangDistribution(int k = 1, double lambda = detail::ONE);

    /** @brief Thread-safe copy constructor. */
    ErlangDistribution(const ErlangDistribution& other);

    /** @brief Copy assignment operator. */
    ErlangDistribution& operator=(const ErlangDistribution& other);

    /** @brief Move constructor. */
    ErlangDistribution(ErlangDistribution&& other) noexcept;

    /** @brief Move assignment operator. */
    ErlangDistribution& operator=(ErlangDistribution&& other) noexcept;

    /** @brief Destructor — defaulted. */
    ~ErlangDistribution() override = default;

    //==========================================================================
    // 2. SAFE FACTORY METHODS (Exception-free construction)
    //==========================================================================

    /**
     * @brief Safely create an Erlang distribution without throwing exceptions.
     * @param k Shape parameter (must be a positive integer)
     * @param lambda Rate parameter (must be positive and finite)
     * @return Result containing either a valid ErlangDistribution or error info
     */
    [[nodiscard]] static Result<ErlangDistribution> create(int k = 1,
                                                           double lambda = detail::ONE) {
        auto v = validateErlangParameters(k, lambda);
        if (v.isError())
            return Result<ErlangDistribution>::makeError(v.errorCode(), v.message());
        return Result<ErlangDistribution>::ok(createUnchecked(k, lambda));
    }

    //==========================================================================
    // 3. PARAMETER GETTERS AND SETTERS
    //==========================================================================

    /** @brief Get shape parameter k. */
    [[nodiscard]] int getK() const noexcept {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        return k_;
    }

    /** @brief Get rate parameter lambda. */
    [[nodiscard]] double getLambda() const noexcept {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        return lambda_;
    }

    /**
     * @brief Set the shape parameter.
     * @param k New shape parameter (must be a positive integer)
     * @throws std::invalid_argument if k < 1
     */
    void setK(int k);

    /**
     * @brief Set the rate parameter.
     * @param lambda New rate parameter (must be positive and finite)
     * @throws std::invalid_argument if lambda is not positive finite
     */
    void setLambda(double lambda);

    /**
     * @brief Set both parameters simultaneously.
     * @throws std::invalid_argument if either parameter is invalid
     */
    void setParameters(int k, double lambda);

    /** @brief Mean = k/lambda — delegates to GammaDistribution. */
    [[nodiscard]] double getMean() const noexcept override { return gamma_.getMean(); }

    /** @brief Variance = k/lambda^2 — delegates to GammaDistribution. */
    [[nodiscard]] double getVariance() const noexcept override { return gamma_.getVariance(); }

    /** @brief Skewness = 2/sqrt(k) — delegates to GammaDistribution. */
    [[nodiscard]] double getSkewness() const noexcept override { return gamma_.getSkewness(); }

    /** @brief Excess kurtosis = 6/k — delegates to GammaDistribution. */
    [[nodiscard]] double getKurtosis() const noexcept override { return gamma_.getKurtosis(); }

    /** @brief Number of parameters (always 2 for Erlang). */
    [[nodiscard]] int getNumParameters() const noexcept override { return 2; }

    /** @brief Distribution name. */
    [[nodiscard]] std::string_view getDistributionName() const noexcept override {
        return "Erlang";
    }

    /** @brief Erlang is continuous. */
    [[nodiscard]] bool isDiscrete() const noexcept override { return false; }

    /** @brief Support lower bound: 0. */
    [[nodiscard]] double getSupportLowerBound() const noexcept override {
        return detail::ZERO_DOUBLE;
    }

    /** @brief Support upper bound: +infinity. */
    [[nodiscard]] double getSupportUpperBound() const noexcept override {
        return std::numeric_limits<double>::infinity();
    }

    //==========================================================================
    // 4. RESULT-BASED SETTERS
    //==========================================================================

    /**
     * @brief Safely set the shape parameter without throwing exceptions.
     * @param k New shape parameter (must be a positive integer)
     */
    [[nodiscard]] VoidResult trySetK(int k) noexcept;

    /**
     * @brief Safely set the rate parameter without throwing exceptions.
     * @param lambda New rate parameter (must be positive and finite)
     */
    [[nodiscard]] VoidResult trySetLambda(double lambda) noexcept;

    /**
     * @brief Safely set both parameters without throwing exceptions.
     */
    [[nodiscard]] VoidResult trySetParameters(int k, double lambda) noexcept;

    /** @brief Check if current parameters are valid. */
    [[nodiscard]] VoidResult validateCurrentParameters() const noexcept;

    //==========================================================================
    // 5. CORE PROBABILITY METHODS
    //==========================================================================

    /**
     * @brief PDF at x. Guards +/-inf and NaN itself before delegating finite
     * x to GammaDistribution — see the class-level "±inf / NaN handling" note.
     */
    [[nodiscard]] double getProbability(double x) const override;

    /**
     * @brief Log-PDF at x. Guards +/-inf and NaN itself before delegating
     * finite x to GammaDistribution — see the class-level note.
     */
    [[nodiscard]] double getLogProbability(double x) const override;

    /** @brief CDF at x — pure delegate to GammaDistribution (already ±inf/NaN-safe). */
    [[nodiscard]] double getCumulativeProbability(double x) const override {
        return gamma_.getCumulativeProbability(x);
    }

    /**
     * @brief Quantile function (inverse CDF) — delegates to GammaDistribution.
     * @param p Probability value in [0, 1]
     * @throws std::invalid_argument if p not in [0, 1]
     */
    [[nodiscard]] double getQuantile(double p) const override { return gamma_.getQuantile(p); }

    /** @brief Generate a single random sample — delegates to GammaDistribution. */
    [[nodiscard]] double sample(std::mt19937& rng) const override { return gamma_.sample(rng); }

    /** @brief Generate multiple random samples — delegates to GammaDistribution. */
    [[nodiscard]] std::vector<double> sample(std::mt19937& rng, size_t n) const override {
        return gamma_.sample(rng, n);
    }

    //==========================================================================
    // 6. DISTRIBUTION MANAGEMENT
    //==========================================================================

    /**
     * @brief Fit k and lambda to data using method of moments.
     *
     * k_hat = round(x_bar^2 / s^2), clamped to >= 1; lambda_hat = k_hat / x_bar.
     *
     * @param values Observed positive data
     * @throws std::invalid_argument if values is empty or contains a
     *         non-positive or non-finite value
     */
    void fit(const std::vector<double>& values) override;

    /** @brief Parallel batch fitting for multiple datasets. */
    static void parallelBatchFit(const std::vector<std::vector<double>>& datasets,
                                 std::vector<ErlangDistribution>& results);

    /** @brief Reset to default parameters (k=1, lambda=1). */
    void reset() noexcept override;

    /** @brief String representation of the distribution. */
    std::string toString() const override;

    //==========================================================================
    // 12. DISTRIBUTION-SPECIFIC UTILITY METHODS
    //==========================================================================

    /** @brief Entropy — delegates to GammaDistribution. */
    [[nodiscard]] double getEntropy() const noexcept override { return gamma_.getEntropy(); }

    /** @brief Median — delegates to GammaDistribution (numerical, via quantile). */
    [[nodiscard]] double getMedian() const override { return gamma_.getMedian(); }

    /** @brief Mode = (k-1)/lambda — delegates to GammaDistribution. */
    [[nodiscard]] double getMode() const noexcept { return gamma_.getMode(); }

    //==========================================================================
    // 13. SMART AUTO-DISPATCH BATCH OPERATIONS
    //==========================================================================
    // For all three overloads below: values and results must have the same
    // size (a mismatch throws std::invalid_argument) and must not overlap (#112).
    // An in-place call silently returns wrong values; overlap is caught only by
    // a debug-mode assert in detail::DispatchUtils. Full contract in
    // core/distribution_interface.h.
    //
    // getProbability/getLogProbability are NOT pure delegates -- see the
    // class-level "±inf / NaN handling" note; they scrub non-finite inputs
    // before delegating and fix up just those output slots afterward.

    /** @brief Batch PDF — ±inf/NaN-guarded dispatch via GammaDistribution. */
    void getProbability(std::span<const double> values, std::span<double> results,
                        const detail::PerformanceHint& hint = {}) const;

    /** @brief Batch log-PDF — ±inf/NaN-guarded dispatch via GammaDistribution. */
    void getLogProbability(std::span<const double> values, std::span<double> results,
                           const detail::PerformanceHint& hint = {}) const;

    /** @brief Batch CDF — pure delegate to GammaDistribution (already ±inf/NaN-safe). */
    void getCumulativeProbability(std::span<const double> values, std::span<double> results,
                                  const detail::PerformanceHint& hint = {}) const {
        gamma_.getCumulativeProbability(values, results, hint);
    }

    //==========================================================================
    // 15. COMPARISON OPERATORS
    //==========================================================================

    bool operator==(const ErlangDistribution& other) const;
    bool operator!=(const ErlangDistribution& other) const;

    //==========================================================================
    // 16. FRIEND FUNCTION STREAM OPERATORS
    //==========================================================================

    friend std::istream& operator>>(std::istream& is, stats::ErlangDistribution&);
    friend std::ostream& operator<<(std::ostream& os, const stats::ErlangDistribution&);

   private:
    //==========================================================================
    // 17. PRIVATE FACTORY METHODS
    //==========================================================================

    /** @brief Create without parameter validation (for internal use). */
    static ErlangDistribution createUnchecked(int k, double lambda) noexcept;

    /** @brief Private bypass-validation constructor. */
    ErlangDistribution(int k, double lambda, bool /*bypassValidation*/) noexcept;

    //==========================================================================
    // 18. PRIVATE BATCH IMPLEMENTATION METHODS
    //==========================================================================

    // Not needed: batch scrub/fixup logic lives directly in the public
    // getProbability/getLogProbability span overloads (src/erlang.cpp).
    // Section retained for template compliance.

    //==========================================================================
    // 19. PRIVATE COMPUTATIONAL METHODS
    //==========================================================================

    /**
     * @brief Sync `gamma_` with current `k_`/`lambda_` and mark cache valid.
     *
     * Calls `gamma_.trySetParameters(k_, lambda_)`, then sets cache_valid_ =
     * true. Called within a held unique_lock on cache_mutex_; acquires
     * gamma_'s own mutex internally — no lock-ordering conflict since gamma_
     * is private.
     */
    void updateCacheUnsafe() const noexcept override;

    /**
     * @brief Validate Erlang parameters, throwing on failure.
     * @param k Shape parameter (must be a positive integer)
     * @param lambda Rate parameter (must be positive and finite)
     * @throws std::invalid_argument if invalid
     */
    static void validateParameters(int k, double lambda) {
        auto v = ::stats::validateErlangParameters(k, lambda);
        if (v.isError())
            throw std::invalid_argument(v.message());
    }

    //==========================================================================
    // 21. DISTRIBUTION PARAMETERS
    //==========================================================================

    /**
     * @brief Shape parameter k — must be a positive integer.
     *
     * Redundant API copy of `gamma_.getAlpha()` (as an int). Exists for O(1)
     * locked reads without entering `gamma_`'s mutex. The invariant
     * `static_cast<double>(k_) == gamma_.getAlpha()` must hold after every
     * setter and `reset()`.
     */
    int k_{1};

    /**
     * @brief Rate parameter lambda — must be positive and finite.
     *
     * Redundant API copy of `gamma_.getBeta()`. The invariant
     * `lambda_ == gamma_.getBeta()` must hold after every setter and `reset()`.
     */
    double lambda_{detail::ONE};

    //==========================================================================
    // 24. PRIVATE DELEGATION MEMBER
    //==========================================================================

    /**
     * @brief Internal Gamma distribution — always maintained as Gamma(k, lambda).
     *
     * Invariant: gamma_.getAlpha() == static_cast<double>(k_) and
     * gamma_.getBeta() == lambda_.
     */
    mutable GammaDistribution gamma_{detail::ONE, detail::ONE};
};

}  // namespace stats

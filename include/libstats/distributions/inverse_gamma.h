#pragma once

// Common distribution includes (consolidates std library and core headers)
#include "libstats/common/distribution_common.h"

// InverseGamma is a reciprocal-transform delegation wrapper over
// GammaDistribution. libstats' GammaDistribution is (alpha SHAPE, beta RATE),
// and InvGamma(alpha, SCALE beta) satisfies 1/X ~ Gamma(alpha, RATE beta), so
// the delegate is GammaDistribution(alpha, beta) with beta passed AS-IS.
// (v2.4.0 kickoff correction on #56: the issue body's `Gamma(alpha, 1/beta)`
// assumed a scale-parameterized Gamma and is wrong for this codebase.)
#include "gamma.h"

namespace stats {

/**
 * @brief Thread-safe Inverse Gamma Distribution: the distribution of 1/X when
 * X is Gamma-distributed.
 *
 * @details The inverse gamma is the conjugate prior for the variance of a
 * normal likelihood with known mean, which is where most of its use comes from.
 *
 * @par Parameterization — beta is a SCALE:
 * `beta` here is the **scale** parameter of the standard inverse-gamma
 * parameterization (identical to scipy's `invgamma(a, scale=beta)`), not a
 * rate. This is the only parameterization in which the density below is
 * correct, and it is what every getter, setter and constructor on this class
 * means by "beta".
 *
 * The delegation is where the distinction earns its keep. libstats'
 * `GammaDistribution` is (alpha shape, beta **RATE**). Because
 *
 *     Y ~ InvGamma(alpha, scale beta)   <=>   1/Y ~ Gamma(alpha, rate beta),
 *
 * the internal delegate is `GammaDistribution(alpha, beta)` with beta passed
 * **unchanged**: this class's SCALE is the delegate's RATE, numerically the
 * same value. No reciprocal is taken on the parameter — only on the variate.
 *
 * @par Mathematical Definition:
 * - PDF:  f(x) = beta^alpha / Gamma(alpha) * x^(-alpha-1) * exp(-beta/x), x > 0
 * - CDF:  F(x) = Q(alpha, beta/x)   (regularized UPPER incomplete gamma)
 * - Parameters: alpha > 0 (shape), beta > 0 (scale)
 * - Support: x in (0, infinity)
 * - Mean:     beta/(alpha-1)                        for alpha > 1, else NaN
 * - Variance: beta^2 / ((alpha-1)^2 (alpha-2))      for alpha > 2, else NaN
 * - Skewness: 4 sqrt(alpha-2)/(alpha-3)             for alpha > 3, else NaN
 * - Excess kurtosis: (30 alpha - 66)/((alpha-3)(alpha-4)) for alpha > 4, else NaN
 * - Mode:     beta/(alpha+1)
 * - Median:   no closed form -- getMedian() returns getQuantile(0.5)
 * - Entropy:  alpha + ln(beta) + lnGamma(alpha) - (1+alpha) psi(alpha)
 *
 * @par CDF derivation, and why it is NOT `1 - delegate.cdf(1/x)` (#49):
 * With Y = 1/X and X ~ Gamma(alpha, rate beta),
 *
 *     P(Y <= x) = P(1/X <= x) = P(X >= 1/x) = Q(alpha, beta/x),
 *
 * because the delegate's CDF is P(alpha, beta*t) and its complement is
 * Q(alpha, beta*t), evaluated here at t = 1/x. This class therefore calls
 * `detail::gamma_q` — the *upper* regularized incomplete gamma, which the math
 * layer already exposes and which computes the continued fraction for Q
 * directly whenever beta/x > alpha+1.
 *
 * The important tail is x -> 0 (the inverse gamma's lower tail, and the one a
 * variance prior actually gets queried in). There beta/x is large, the CDF is
 * tiny, `1 - CDF_Gamma(1/x)` would be `1 - (1 - tiny)` and would return
 * literally zero once the CDF rounds to 1 — while `gamma_q` returns the tiny
 * value with full relative precision from its own continued fraction. The
 * complement is never formed. `GammaDistribution`'s public API exposes only
 * the lower tail, so the call goes one layer down to the same routine the
 * delegate itself uses; that is the whole reason for the layering exception.
 *
 * Conversely the survival function 1 - F(x) = P(alpha, beta/x) is available
 * from `detail::gamma_p` with the same property in the other direction, and is
 * exposed as getSurvivalProbability().
 *
 * @par PDF / LogPDF — delegated through the reciprocal:
 * logpdf(x) = delegate.logpdf(1/x) - 2 ln x, exact in log space (the Jacobian
 * of Y = 1/X is 1/x^2). PDF is exp() of that rather than
 * delegate.pdf(1/x)/x^2, so no intermediate overflows at tiny x where the
 * delegate's density is astronomically large but the inverse-gamma density is
 * not. The transform layer supplies its own +/-inf, 1/0 and NaN handling: the
 * reciprocal maps the interesting edges onto each other (x -> 0+ gives
 * 1/x -> +inf, x -> +inf gives 1/x -> 0+), and for x below ~5.6e-309 the
 * reciprocal overflows outright, so none of those cases may be handed to the
 * delegate.
 *
 * @par +/-inf and NaN handling (#103):
 * pdf(+/-inf) = 0, logpdf(+/-inf) = -inf, cdf(-inf) = 0, cdf(+inf) = 1, NaN
 * propagates. At and below the support edge (x <= 0) pdf = 0, logpdf = -inf,
 * cdf = 0 — the density's exp(-beta/x) factor sends it to zero faster than any
 * power, so 0 is the true limit for every alpha. Scalar and batch agree
 * element for element.
 *
 * @par Quantile accuracy (#104):
 * Safeguarded bisection in log-x against the CDF (p <= 1/2) or the survival
 * function (p > 1/2), so the residual is evaluated on the small side in both
 * tails; 80 halvings of [e^-745, e^709] close the bracket to below one ulp.
 * Never NaN for p in (0,1); +infinity only when the true quantile exceeds the
 * double range. This is preferred over `1 / delegate.getQuantile(1-p)` because
 * the delegate's solver stops on an absolute residual and would be solving at
 * q = 1-p near 1 exactly where its own accuracy is worst.
 *
 * @par Achievable accuracy — the law, not a budget:
 * Following the #49 pattern, what is stated is what this layer can deliver,
 * and the tests are gated to it rather than to a flat figure the shared math
 * layer cannot reach.
 * - **PDF / LogPDF**: full double accuracy, < 1e-13 relative against 50-dps
 *   mpmath. The delegate's log-space formula plus an exact Jacobian; nothing
 *   cancels.
 * - **CDF and survival**: bounded by `detail::gamma_q` / `detail::gamma_p`,
 *   whose series and continued fraction both stop on a relative residual of
 *   `detail::DEFAULT_TOLERANCE` (1e-8). The binding case is beta/x ~ alpha,
 *   right at their series/continued-fraction switch: measured worst case over
 *   the test grid is 1.04e-8 relative, at InvGamma(50,3) CDF(0.06) where
 *   beta/x = 50 and alpha = 50. Away from that switch the error is orders of
 *   magnitude smaller -- the extreme lower tail measures 1.8e-15 relative at
 *   InvGamma(3,2) CDF(0.05) and 2.6e-15 at CDF(0.02), where the true values
 *   are 3.6e-15 and 1.9e-40. This is a property of the shared math layer
 *   and is *not* what the upper-incomplete-gamma formulation buys: that
 *   formulation is what stops the lower tail from collapsing to exactly 0,
 *   which is a total loss rather than a 1e-8 one.
 * - **Quantile**: divides the CDF's relative error by the local tail
 *   elasticity |d ln F / d ln x| (alpha in the lower tail), so it is tighter
 *   than the CDF driving it. Measured 1e-16 to 3e-12 relative from p = 1e-12
 *   to p = 1 - 1e-12.
 * - **Quantile at p near 1 is limited by the caller's argument, not by this
 *   code**: a double p in [1/2, 1) carries its complement only to
 *   ulp(1)/2 = 1.11e-16 absolute. `1.0 - p` is exact by Sterbenz's lemma, so
 *   the solver reaches the complement it was handed to full precision, but
 *   that complement is not the decimal the caller wrote. Work from
 *   getSurvivalProbability() for the far upper tail.
 * - **Entropy** is limited by `detail::digamma`, not by this class: measured
 *   ~1e-8 absolute.
 * - **Batch vs scalar**: the CDF path is this class's own scalar kernel on
 *   every dispatch tier, so batch equals scalar bit for bit. PDF/LogPDF go
 *   through the Gamma delegate's batch API, whose vectorized kernel differs
 *   from its own scalar path in the last ulp once the batch clears the
 *   dispatch threshold.
 *
 * @par MLE:
 * Gamma's MLE structure applied to the reciprocals: 1/x_i are i.i.d.
 * Gamma(alpha, rate beta), so `fit()` inverts the data, fits a temporary
 * GammaDistribution to it, and lifts the estimates back unchanged
 * (alpha_hat = shape, beta_hat = the fitted RATE, which is this class's SCALE).
 *
 * @par Thread Safety:
 * All methods are thread-safe via the reader-writer lock in DistributionBase
 * plus atomic parameter copies for the lock-free fast path.
 *
 * @author libstats Development Team
 * @version 2.4.0
 * @since 2.4.0
 */
class InverseGammaDistribution : public DistributionBase {
   public:
    // Dispatch metadata
    static constexpr detail::DistributionType kDistributionType =
        detail::DistributionType::INVERSE_GAMMA;
    static constexpr bool kIsDiscrete = false;

   public:
    //==========================================================================
    // 1. CONSTRUCTORS AND DESTRUCTOR
    //==========================================================================

    /**
     * @brief Constructs an inverse gamma distribution.
     * @param alpha Shape parameter (positive and finite, default: 1)
     * @param beta  SCALE parameter (positive and finite, default: 1)
     * @throws std::invalid_argument if either parameter is not positive finite
     */
    explicit InverseGammaDistribution(double alpha = detail::ONE, double beta = detail::ONE);

    /** @brief Thread-safe copy constructor. */
    InverseGammaDistribution(const InverseGammaDistribution& other);

    /** @brief Copy assignment operator. */
    InverseGammaDistribution& operator=(const InverseGammaDistribution& other);

    /** @brief Move constructor. */
    InverseGammaDistribution(InverseGammaDistribution&& other) noexcept;

    /** @brief Move assignment operator. */
    InverseGammaDistribution& operator=(InverseGammaDistribution&& other) noexcept;

    /** @brief Destructor — defaulted. */
    ~InverseGammaDistribution() override = default;

    //==========================================================================
    // 2. SAFE FACTORY METHODS (Exception-free construction)
    //==========================================================================

    /**
     * @brief Safely create an inverse gamma distribution without throwing.
     * @param alpha Shape parameter (must be positive and finite)
     * @param beta  SCALE parameter (must be positive and finite)
     */
    [[nodiscard]] static Result<InverseGammaDistribution> create(
        double alpha = detail::ONE, double beta = detail::ONE) noexcept {
        auto v = validateInverseGammaParameters(alpha, beta);
        if (v.isError())
            return Result<InverseGammaDistribution>::makeError(v.errorCode(), v.message());
        return Result<InverseGammaDistribution>::ok(createUnchecked(alpha, beta));
    }

    //==========================================================================
    // 3. PARAMETER GETTERS AND SETTERS
    //==========================================================================

    /** @brief Get shape parameter alpha. */
    [[nodiscard]] double getAlpha() const noexcept {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        return alpha_;
    }

    /** @brief Get SCALE parameter beta (see the class note on parameterization). */
    [[nodiscard]] double getBeta() const noexcept {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        return beta_;
    }

    /** @brief Lock-free read of alpha (falls back to the locked path if stale). */
    [[nodiscard]] double getAlphaAtomic() const noexcept;

    /** @brief Lock-free read of beta (falls back to the locked path if stale). */
    [[nodiscard]] double getBetaAtomic() const noexcept;

    /**
     * @brief Set the shape parameter.
     * @throws std::invalid_argument if alpha is not positive finite
     */
    void setAlpha(double alpha);

    /**
     * @brief Set the SCALE parameter.
     * @throws std::invalid_argument if beta is not positive finite
     */
    void setBeta(double beta);

    /**
     * @brief Set both parameters simultaneously.
     * @throws std::invalid_argument if either parameter is invalid
     */
    void setParameters(double alpha, double beta);

    /** @brief Mean = beta/(alpha-1) for alpha > 1; NaN otherwise (undefined). */
    [[nodiscard]] double getMean() const override;

    /** @brief Variance for alpha > 2; NaN otherwise (undefined). */
    [[nodiscard]] double getVariance() const override;

    /** @brief Skewness = 4 sqrt(alpha-2)/(alpha-3) for alpha > 3; NaN otherwise. */
    [[nodiscard]] double getSkewness() const override;

    /** @brief Excess kurtosis for alpha > 4; NaN otherwise (undefined). */
    [[nodiscard]] double getKurtosis() const override;

    /** @brief Number of parameters (always 2). */
    [[nodiscard]] int getNumParameters() const noexcept override { return 2; }

    /** @brief Distribution name. */
    [[nodiscard]] std::string_view getDistributionName() const noexcept override {
        return "InverseGamma";
    }

    /** @brief Inverse gamma is continuous. */
    [[nodiscard]] bool isDiscrete() const noexcept override { return false; }

    /** @brief Support lower bound: 0 (the density is 0 there, support is open). */
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

    /** @brief Safely set alpha without throwing exceptions. */
    [[nodiscard]] VoidResult trySetAlpha(double alpha) noexcept;

    /** @brief Safely set beta (SCALE) without throwing exceptions. */
    [[nodiscard]] VoidResult trySetBeta(double beta) noexcept;

    /** @brief Safely set both parameters without throwing exceptions. */
    [[nodiscard]] VoidResult trySetParameters(double alpha, double beta) noexcept;

    /** @brief Check if current parameters are valid. */
    [[nodiscard]] VoidResult validateCurrentParameters() const noexcept;

    //==========================================================================
    // 5. CORE PROBABILITY METHODS
    //==========================================================================

    /** @brief PDF at x = exp(getLogProbability(x)). Guards +/-inf, 1/0 and NaN. */
    [[nodiscard]] double getProbability(double x) const override;

    /**
     * @brief Log-PDF at x = delegate.logpdf(1/x) - 2 ln x. Guards +/-inf, the
     * reciprocal's own overflow at x < ~5.6e-309, and NaN (#103).
     */
    [[nodiscard]] double getLogProbability(double x) const override;

    /**
     * @brief CDF at x = Q(alpha, beta/x), the regularized UPPER incomplete
     * gamma — never `1 - CDF_Gamma(1/x)`. See the class-level derivation.
     */
    [[nodiscard]] double getCumulativeProbability(double x) const override;

    /**
     * @brief Survival function 1 - CDF(x) = P(alpha, beta/x), computed directly
     * from the *lower* regularized incomplete gamma rather than by subtracting
     * the CDF from one, so it keeps full relative precision in the upper tail.
     */
    [[nodiscard]] double getSurvivalProbability(double x) const;

    /**
     * @brief Quantile function (inverse CDF).
     * @param p Probability in [0, 1]
     * @throws std::invalid_argument if p is NaN or outside [0, 1]
     * @see the class-level "Quantile accuracy" note.
     */
    [[nodiscard]] double getQuantile(double p) const override;

    /** @brief Single random sample: 1 / delegate.sample(rng). */
    [[nodiscard]] double sample(std::mt19937& rng) const override;

    /** @brief Multiple random samples. */
    [[nodiscard]] std::vector<double> sample(std::mt19937& rng, size_t n) const override;

    //==========================================================================
    // 6. DISTRIBUTION MANAGEMENT
    //==========================================================================

    /**
     * @brief Fit alpha and beta by applying Gamma's MLE to the reciprocals.
     * @param values Observed positive data
     * @throws std::invalid_argument if values is empty or holds a non-positive
     *         or non-finite value, or if any reciprocal overflows
     */
    void fit(const std::vector<double>& values) override;

    /** @brief Parallel batch fitting for multiple datasets. */
    static void parallelBatchFit(const std::vector<std::vector<double>>& datasets,
                                 std::vector<InverseGammaDistribution>& results);

    /** @brief Reset to default parameters (alpha = 1, beta = 1). */
    void reset() noexcept override;

    /** @brief String representation of the distribution. */
    std::string toString() const override;

    //==========================================================================
    // 12. DISTRIBUTION-SPECIFIC UTILITY METHODS
    //==========================================================================

    /**
     * @brief Differential entropy in nats:
     * alpha + ln(beta) + lnGamma(alpha) - (1+alpha) psi(alpha).
     * Verified against numerical quadrature to 12 digits for (3,2), (50,3)
     * and (0.5,1).
     */
    [[nodiscard]] double getEntropy() const override;

    /**
     * @brief Median. No closed form exists, so this returns getQuantile(0.5)
     * (documented deviation — every other moment here is a closed form).
     */
    [[nodiscard]] double getMedian() const override { return getQuantile(detail::HALF); }

    /** @brief Mode = beta/(alpha+1); always defined. */
    [[nodiscard]] double getMode() const;

    //==========================================================================
    // 13. SMART AUTO-DISPATCH BATCH OPERATIONS
    //==========================================================================
    // For all three overloads: values and results must have the same size (a
    // mismatch throws std::invalid_argument) and must not overlap (#112). An
    // in-place call silently returns wrong values; overlap is caught only by a
    // debug-mode assert in detail::DispatchUtils. Full contract in
    // core/distribution_interface.h.
    //
    // getProbability/getLogProbability build a scratch buffer of reciprocals,
    // hand that to the Gamma delegate's batch API, and then apply the -2 ln x
    // Jacobian and the edge fixups from the *caller's input span*. Re-reading
    // `values` after `results` has been written is exactly what the
    // non-overlap contract exists to permit (#112, same shape as the Gaussian
    // and Gamma CDF tail fixups). The delegate is never called with its input
    // and output aliased: the scratch buffer is a distinct allocation.
    //
    // getCumulativeProbability does not delegate at all — detail::gamma_q has
    // no vector form, so it runs its own scalar kernel under autoDispatch.

    /** @brief Batch PDF — reciprocal transform, then the Gamma delegate. */
    void getProbability(std::span<const double> values, std::span<double> results,
                        const detail::PerformanceHint& hint = {}) const;

    /** @brief Batch log-PDF — reciprocal transform, then the Gamma delegate. */
    void getLogProbability(std::span<const double> values, std::span<double> results,
                           const detail::PerformanceHint& hint = {}) const;

    /** @brief Batch CDF — own scalar kernel over detail::gamma_q. */
    void getCumulativeProbability(std::span<const double> values, std::span<double> results,
                                  const detail::PerformanceHint& hint = {}) const;

    //==========================================================================
    // 15. COMPARISON OPERATORS
    //==========================================================================

    bool operator==(const InverseGammaDistribution& other) const;
    bool operator!=(const InverseGammaDistribution& other) const;

    //==========================================================================
    // 16. FRIEND FUNCTION STREAM OPERATORS
    //==========================================================================

    friend std::istream& operator>>(std::istream& is, stats::InverseGammaDistribution&);
    friend std::ostream& operator<<(std::ostream& os, const stats::InverseGammaDistribution&);

   private:
    //==========================================================================
    // 17. PRIVATE FACTORY METHODS
    //==========================================================================

    /** @brief Create without parameter validation (for internal use). */
    static InverseGammaDistribution createUnchecked(double alpha, double beta) noexcept;

    /** @brief Private bypass-validation constructor. */
    InverseGammaDistribution(double alpha, double beta, bool /*bypassValidation*/) noexcept;

    //==========================================================================
    // 18. PRIVATE BATCH IMPLEMENTATION METHODS
    //==========================================================================

    /** @brief Scalar CDF kernel over a raw range, from a cache snapshot. */
    static void cdfKernel(const double* values, double* results, std::size_t count, double alpha,
                          double beta) noexcept;

    //==========================================================================
    // 19. PRIVATE COMPUTATIONAL METHODS
    //==========================================================================

    /**
     * @brief Classify x for the reciprocal transform.
     *
     * Returns true when x is an ordinary interior point whose reciprocal is a
     * finite positive double, so the Gamma delegate may be called with it. When
     * it returns false, @p edge_logpdf holds the correct log-density for x and
     * the delegate must be bypassed entirely.
     */
    [[nodiscard]] static bool reciprocalIsUsable(double x, double& inv_x,
                                                 double& edge_logpdf) noexcept;

    /** @brief CDF from an explicit cache snapshot (shared by scalar/batch). */
    [[nodiscard]] static double cdfImpl(double x, double alpha, double beta) noexcept;

    /** @brief Survival from an explicit cache snapshot. */
    [[nodiscard]] static double sfImpl(double x, double alpha, double beta) noexcept;

    /**
     * @brief Resync `gamma_` with alpha_/beta_ and mark the cache valid.
     * Called with a held unique_lock on cache_mutex_.
     */
    void updateCacheUnsafe() const noexcept override;

    //==========================================================================
    // 21. PRIVATE VALIDATION METHODS
    //==========================================================================

    /** @brief Validate parameters, throwing on failure. */
    static void validateParameters(double alpha, double beta) {
        auto v = ::stats::validateInverseGammaParameters(alpha, beta);
        if (v.isError())
            throw std::invalid_argument(v.message());
    }

    //==========================================================================
    // 23. DISTRIBUTION PARAMETERS
    //==========================================================================

    /** @brief Shape parameter alpha — positive and finite. */
    double alpha_{detail::ONE};

    /**
     * @brief SCALE parameter beta — positive and finite.
     *
     * Redundant API copy of `gamma_.getBeta()`, which is a RATE. The two are
     * numerically equal by the reciprocal identity documented on the class;
     * the invariant `beta_ == gamma_.getBeta()` must hold after every setter
     * and reset().
     */
    double beta_{detail::ONE};

    /** @brief Atomic copies for lock-free parameter access. */
    mutable std::atomic<double> atomicAlpha_{detail::ONE};
    mutable std::atomic<double> atomicBeta_{detail::ONE};
    mutable std::atomic<bool> atomicParamsValid_{false};

    //==========================================================================
    // 25. PRIVATE DELEGATION MEMBER
    //==========================================================================

    /**
     * @brief Internal Gamma delegate — always Gamma(shape alpha_, RATE beta_).
     *
     * Invariant: gamma_.getAlpha() == alpha_ and gamma_.getBeta() == beta_.
     * Supplies PDF/LogPDF (through the reciprocal), sampling and MLE. The CDF
     * and quantile deliberately bypass it — see the class-level notes.
     */
    mutable GammaDistribution gamma_{detail::ONE, detail::ONE};
};

}  // namespace stats

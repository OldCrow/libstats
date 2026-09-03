#pragma once

// Common distribution includes (consolidates std library and core headers)
#include "libstats/common/distribution_common.h"

// F (Fisher-Snedecor) is a transform wrapper over BetaDistribution:
//   Y ~ Beta(d1/2, d2/2)  <=>  X = (d2/d1) * Y/(1-Y) ~ F(d1, d2).
// The Beta delegate is held and kept in sync as Beta(d1/2, d2/2); it provides
// sampling. PDF/LogPDF/CDF/Quantile are NOT routed through the delegate's
// public API -- see the "Why not a pure delegation" note on the class.
#include "beta.h"

namespace stats {

/**
 * @brief Thread-safe F (Fisher-Snedecor) Distribution: the ratio of two
 * independent chi-squared variates, each divided by its degrees of freedom.
 *
 * @details The F distribution is the null distribution of the variance-ratio
 * statistic in ANOVA and regression F-tests. If U ~ chi2(d1) and V ~ chi2(d2)
 * are independent, then X = (U/d1)/(V/d2) ~ F(d1, d2).
 *
 * @par Mathematical Definition:
 * - PDF:  f(x) = [ (d1 x)^(d1/2) d2^(d2/2) ] /
 *                [ x * (d1 x + d2)^((d1+d2)/2) * B(d1/2, d2/2) ],  x > 0
 * - CDF:  F(x) = I_y(d1/2, d2/2),  y = d1 x / (d1 x + d2)
 * - Parameters: d1 > 0, d2 > 0 (degrees of freedom, real-valued)
 * - Support: x in [0, infinity)
 * - Mean:     d2/(d2-2)                          for d2 > 2,  else NaN
 * - Variance: 2 d2^2 (d1+d2-2) / (d1 (d2-2)^2 (d2-4))   for d2 > 4,  else NaN
 * - Skewness: (2d1+d2-2) sqrt(8(d2-4)) /
 *             ((d2-6) sqrt(d1) sqrt(d1+d2-2))    for d2 > 6,  else NaN
 * - Excess kurtosis: 12 [d1(5d2-22)(d1+d2-2) + (d2-4)(d2-2)^2] /
 *                    [d1 (d2-6)(d2-8)(d1+d2-2)]  for d2 > 8,  else NaN
 * - Mode:     ((d1-2)/d1) * (d2/(d2+2))          for d1 > 2,  else 0
 * - Median:   no closed form -- getMedian() returns getQuantile(0.5)
 *
 * @par Relationship to Beta:
 * With a = d1/2, b = d2/2 and the change of variable
 *   y = d1 x / (d1 x + d2),   x = (d2/d1) * y / (1 - y),
 * X ~ F(d1,d2) exactly when Y ~ Beta(a, b). `beta_` is held as Beta(a, b) and
 * supplies `sample()`.
 *
 * @par Why not a pure delegation (the #49 lesson):
 * Both the Beta PDF and the Beta CDF need the complement `1 - y`. As x grows,
 * y -> 1 and forming `1 - y` by subtraction destroys every significant bit of
 * the complement, while the complement has an exact direct form
 *
 *     ybar = d2 / (d1 x + d2)        (never  1 - y)
 *
 * that costs one division. `BetaDistribution`'s public API takes y and
 * recomputes `1 - y` internally, so it cannot be steered onto the accurate
 * side. Therefore:
 *
 * - **PDF / LogPDF** use the closed form above directly (log space, then exp).
 *   No subtraction of near-equal quantities appears anywhere in it.
 * - **CDF** calls `detail::beta_i` (the same regularized incomplete beta that
 *   `BetaDistribution` itself calls -- one layer down, not a new algorithm) and
 *   chooses the branch so the incomplete-beta argument is always the *small*
 *   one, using the symmetry I_y(a,b) = 1 - I_ybar(b,a):
 *
 *       y < (a+1)/(a+b+2)  ->  CDF = I_y(a, b)          [y computed directly]
 *       otherwise          ->  CDF = 1 - I_ybar(b, a)   [ybar computed directly]
 *
 *   The branch condition is exactly `detail::beta_i`'s own internal
 *   series/complement switch, so the routine always takes its direct
 *   continued-fraction path and never has to form the complement itself.
 * - **Survival** (used internally by the quantile solver) is
 *   `I_ybar(b, a)` -- computed, never `1 - CDF`.
 *
 * Note that `detail::f_cdf` in math_utils.h is the older, unsteered form
 * (`beta_i(y, a, b)` with y formed directly); it is left untouched and is not
 * used here.
 *
 * @par Quantile accuracy (#104):
 * `getQuantile(p)` is a safeguarded bisection in log-x against the steered CDF
 * (for p <= 1/2) or the steered survival function (for p > 1/2), so the
 * residual is always evaluated on the small side of the probability scale.
 * 80 halvings of the full double exponent range [e^-745, e^709] bring the
 * bracket below one ulp of the answer. This is deliberately *not*
 * `detail::inverse_beta_i`: that routine stops on an **absolute** residual
 * (`detail::DEFAULT_TOLERANCE` = 1e-8) from a start point clamped to
 * [1e-8, 1-1e-8], so for p below ~1e-8 it can return its clamp unchanged.
 * The result is never NaN for p in (0,1); it is +infinity only when the true
 * quantile exceeds the double range, and 0 only when it falls below it.
 *
 * @par Achievable accuracy — the law, not a budget:
 * Following the #49 pattern, what is stated here is what this layer can
 * actually deliver, and the tests are gated to it rather than to a flat figure
 * the shared math layer cannot reach.
 * - **PDF / LogPDF**: full double accuracy. The closed form contains no
 *   cancellation at all. Measured < 1e-13 relative against 50-dps mpmath
 *   across the whole test grid.
 * - **CDF and survival**: bounded by `detail::beta_i`'s continued fraction,
 *   which stops on |delta - 1| < `detail::DEFAULT_TOLERANCE` (1e-8). Measured
 *   worst case over the test grid (d1 and d2 from 1 to 200) is 4.3e-10
 *   relative, and it sits in the *central* region (F(100,200) at x = 1), where
 *   the continued fraction needs the most terms. The extreme tails are far
 *   tighter -- 2e-15 to 7e-15 relative at CDF(1e-6) and SF(1e6) for F(5,10) --
 *   because a small incomplete-beta argument converges almost immediately.
 *   All of this is a property of the shared math layer, equal on both
 *   branches, and is *not* what the tail steering fixes: the steering is what
 *   stops the upper tail from collapsing to exactly 0 or 1, which is a total
 *   loss rather than a 1e-10 one.
 * - **Quantile**: inherits the CDF's relative error divided by the local tail
 *   elasticity |d ln F / d ln x|, so it is better than the CDF rather than
 *   worse. Measured 1e-15 to 2e-13 relative from p = 1e-12 up to 1 - 1e-12.
 * - **Quantile at p near 1 is limited by the caller's argument, not by this
 *   code.** A double p in [1/2, 1) carries its own complement only to
 *   ulp(1)/2 = 1.11e-16 absolute, so the 1-p that reaches any implementation
 *   has relative error 1.11e-16/(1-p) — 2.2e-4 at p = 1 - 1e-12. Divided by
 *   the upper-tail elasticity d2/2, that is the induced relative error in the
 *   returned quantile (4.4e-6 for F(5,10) at that p). The subtraction `1.0 - p`
 *   is itself exact here by Sterbenz's lemma, so the solver reaches the
 *   complement it was handed to full precision; the information was destroyed
 *   before the call. Callers who need the far upper tail should work from
 *   getSurvivalProbability(), never from a p near 1 or from 1 - CDF.
 *
 * @par +/-inf and NaN handling (#103):
 * Every entry point guards non-finite input itself rather than inheriting a
 * delegate's behaviour: pdf(+/-inf) = 0, logpdf(+/-inf) = -inf, cdf(-inf) = 0,
 * cdf(+inf) = 1, and NaN propagates to NaN. Below the support, pdf = 0,
 * logpdf = -inf, cdf = 0. At the support edge x = 0 the mathematical limit is
 * returned: +inf density for d1 < 2, a finite density for d1 == 2, and 0 for
 * d1 > 2. Scalar and batch paths agree element-for-element.
 *
 * @par MLE:
 * Method of moments, as a documented best-effort: an F distribution's degrees
 * of freedom are normally fixed by the experimental design, not estimated, and
 * no closed-form MLE exists. `fit()` inverts the mean to get d2 and then the
 * variance to get d1, clamping both into a range where the moment equations
 * are defined, and falling back to documented defaults when the sample moments
 * do not admit a solution. See src/fisher_f.cpp for the exact fallbacks.
 *
 * @par Thread Safety:
 * All methods are thread-safe via the reader-writer lock in DistributionBase
 * plus atomic parameter copies for the lock-free fast path.
 *
 * @author libstats Development Team
 * @version 2.4.0
 * @since 2.4.0
 */
class FDistribution : public DistributionBase {
   public:
    // Dispatch metadata
    static constexpr detail::DistributionType kDistributionType =
        detail::DistributionType::FISHER_F;
    static constexpr bool kIsDiscrete = false;

   public:
    //==========================================================================
    // 1. CONSTRUCTORS AND DESTRUCTOR
    //==========================================================================

    /**
     * @brief Constructs an F distribution with the given degrees of freedom.
     * @param d1 Numerator degrees of freedom (positive and finite, default: 1)
     * @param d2 Denominator degrees of freedom (positive and finite, default: 1)
     * @throws std::invalid_argument if either parameter is not positive finite
     */
    explicit FDistribution(double d1 = detail::ONE, double d2 = detail::ONE);

    /** @brief Thread-safe copy constructor. */
    FDistribution(const FDistribution& other);

    /** @brief Copy assignment operator. */
    FDistribution& operator=(const FDistribution& other);

    /** @brief Move constructor. */
    FDistribution(FDistribution&& other) noexcept;

    /** @brief Move assignment operator. */
    FDistribution& operator=(FDistribution&& other) noexcept;

    /** @brief Destructor — defaulted. */
    ~FDistribution() override = default;

    //==========================================================================
    // 2. SAFE FACTORY METHODS (Exception-free construction)
    //==========================================================================

    /**
     * @brief Safely create an F distribution without throwing exceptions.
     * @param d1 Numerator degrees of freedom (must be positive and finite)
     * @param d2 Denominator degrees of freedom (must be positive and finite)
     * @return Result containing either a valid FDistribution or error info
     */
    [[nodiscard]] static Result<FDistribution> create(double d1 = detail::ONE,
                                                      double d2 = detail::ONE) noexcept {
        auto v = validateFisherFParameters(d1, d2);
        if (v.isError())
            return Result<FDistribution>::makeError(v.errorCode(), v.message());
        return Result<FDistribution>::ok(createUnchecked(d1, d2));
    }

    //==========================================================================
    // 3. PARAMETER GETTERS AND SETTERS
    //==========================================================================

    /** @brief Get numerator degrees of freedom d1. */
    [[nodiscard]] double getD1() const noexcept {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        return d1_;
    }

    /** @brief Get denominator degrees of freedom d2. */
    [[nodiscard]] double getD2() const noexcept {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        return d2_;
    }

    /** @brief Lock-free read of d1 (falls back to the locked path if stale). */
    [[nodiscard]] double getD1Atomic() const noexcept;

    /** @brief Lock-free read of d2 (falls back to the locked path if stale). */
    [[nodiscard]] double getD2Atomic() const noexcept;

    /**
     * @brief Set the numerator degrees of freedom.
     * @throws std::invalid_argument if d1 is not positive finite
     */
    void setD1(double d1);

    /**
     * @brief Set the denominator degrees of freedom.
     * @throws std::invalid_argument if d2 is not positive finite
     */
    void setD2(double d2);

    /**
     * @brief Set both degrees of freedom simultaneously.
     * @throws std::invalid_argument if either parameter is invalid
     */
    void setParameters(double d1, double d2);

    /** @brief Mean = d2/(d2-2) for d2 > 2; NaN otherwise (undefined). */
    [[nodiscard]] double getMean() const override;

    /** @brief Variance for d2 > 4; NaN otherwise (undefined). */
    [[nodiscard]] double getVariance() const override;

    /** @brief Skewness for d2 > 6; NaN otherwise (undefined). */
    [[nodiscard]] double getSkewness() const override;

    /** @brief Excess kurtosis for d2 > 8; NaN otherwise (undefined). */
    [[nodiscard]] double getKurtosis() const override;

    /** @brief Number of parameters (always 2 for F). */
    [[nodiscard]] int getNumParameters() const noexcept override { return 2; }

    /** @brief Distribution name. */
    [[nodiscard]] std::string_view getDistributionName() const noexcept override {
        return "FisherF";
    }

    /** @brief F is continuous. */
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

    /** @brief Safely set d1 without throwing exceptions. */
    [[nodiscard]] VoidResult trySetD1(double d1) noexcept;

    /** @brief Safely set d2 without throwing exceptions. */
    [[nodiscard]] VoidResult trySetD2(double d2) noexcept;

    /** @brief Safely set both parameters without throwing exceptions. */
    [[nodiscard]] VoidResult trySetParameters(double d1, double d2) noexcept;

    /** @brief Check if current parameters are valid. */
    [[nodiscard]] VoidResult validateCurrentParameters() const noexcept;

    //==========================================================================
    // 5. CORE PROBABILITY METHODS
    //==========================================================================

    /**
     * @brief PDF at x. Closed form in log space, then exp — see the class-level
     * "Why not a pure delegation" note. Guards +/-inf and NaN (#103).
     */
    [[nodiscard]] double getProbability(double x) const override;

    /** @brief Log-PDF at x. Closed form; guards +/-inf and NaN (#103). */
    [[nodiscard]] double getLogProbability(double x) const override;

    /**
     * @brief CDF at x = I_y(d1/2, d2/2), evaluated on the small side of the
     * incomplete-beta argument — see the class-level note. Guards +/-inf/NaN.
     */
    [[nodiscard]] double getCumulativeProbability(double x) const override;

    /**
     * @brief Survival function 1 - CDF(x), computed as I_ybar(d2/2, d1/2) with
     * ybar = d2/(d1 x + d2) formed directly.
     *
     * This is the accurate upper tail: it never subtracts the CDF from one, so
     * it retains full relative precision for probabilities far below 1e-16
     * where `1 - getCumulativeProbability(x)` would return exactly 0.
     */
    [[nodiscard]] double getSurvivalProbability(double x) const;

    /**
     * @brief Quantile function (inverse CDF).
     * @param p Probability in [0, 1]
     * @throws std::invalid_argument if p is NaN or outside [0, 1]
     * @see the class-level "Quantile accuracy" note for the algorithm and its
     *      measured accuracy.
     */
    [[nodiscard]] double getQuantile(double p) const override;

    /** @brief Generate a single random sample (via the Beta delegate). */
    [[nodiscard]] double sample(std::mt19937& rng) const override;

    /** @brief Generate multiple random samples. */
    [[nodiscard]] std::vector<double> sample(std::mt19937& rng, size_t n) const override;

    //==========================================================================
    // 6. DISTRIBUTION MANAGEMENT
    //==========================================================================

    /**
     * @brief Fit d1 and d2 to data by method of moments (best effort).
     * @param values Observed positive data
     * @throws std::invalid_argument if values is empty or holds a non-positive
     *         or non-finite value
     * @see the class-level MLE note; src/fisher_f.cpp documents every fallback.
     */
    void fit(const std::vector<double>& values) override;

    /** @brief Parallel batch fitting for multiple datasets. */
    static void parallelBatchFit(const std::vector<std::vector<double>>& datasets,
                                 std::vector<FDistribution>& results);

    /** @brief Reset to default parameters (d1 = 1, d2 = 1). */
    void reset() noexcept override;

    /** @brief String representation of the distribution. */
    std::string toString() const override;

    //==========================================================================
    // 12. DISTRIBUTION-SPECIFIC UTILITY METHODS
    //==========================================================================

    /**
     * @brief Differential entropy in nats.
     *
     * H = ln B(a,b) + ln(d2/d1) - (a-1) psi(a) - (b+1) psi(b) + (a+b) psi(a+b),
     * with a = d1/2, b = d2/2. Derived from H(Beta(a,b)) plus the Jacobian term
     * E[ln|dx/dy|] = ln(d2/d1) - 2 E[ln(1-Y)] = ln(d2/d1) - 2(psi(b)-psi(a+b));
     * verified against numerical quadrature to 12 digits for (5,10), (3,7),
     * (100,200) and (2,30).
     */
    [[nodiscard]] double getEntropy() const override;

    /**
     * @brief Median. The F distribution has no closed-form median, so this
     * returns getQuantile(0.5) (documented deviation — every other moment on
     * this class is a closed form).
     */
    [[nodiscard]] double getMedian() const override { return getQuantile(detail::HALF); }

    /** @brief Mode = ((d1-2)/d1)(d2/(d2+2)) for d1 > 2; 0 otherwise. */
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
    // Every kernel here reproduces the scalar path element-for-element,
    // including the #103 non-finite guards, so batch == scalar exactly.

    /** @brief Batch PDF with automatic strategy selection. */
    void getProbability(std::span<const double> values, std::span<double> results,
                        const detail::PerformanceHint& hint = {}) const;

    /** @brief Batch log-PDF with automatic strategy selection. */
    void getLogProbability(std::span<const double> values, std::span<double> results,
                           const detail::PerformanceHint& hint = {}) const;

    /** @brief Batch CDF with automatic strategy selection. */
    void getCumulativeProbability(std::span<const double> values, std::span<double> results,
                                  const detail::PerformanceHint& hint = {}) const;

    //==========================================================================
    // 15. COMPARISON OPERATORS
    //==========================================================================

    bool operator==(const FDistribution& other) const;
    bool operator!=(const FDistribution& other) const;

    //==========================================================================
    // 16. FRIEND FUNCTION STREAM OPERATORS
    //==========================================================================

    friend std::istream& operator>>(std::istream& is, stats::FDistribution&);
    friend std::ostream& operator<<(std::ostream& os, const stats::FDistribution&);

   private:
    //==========================================================================
    // 17. PRIVATE FACTORY METHODS
    //==========================================================================

    /** @brief Create without parameter validation (for internal use). */
    static FDistribution createUnchecked(double d1, double d2) noexcept;

    /** @brief Private bypass-validation constructor. */
    FDistribution(double d1, double d2, bool /*bypassValidation*/) noexcept;

    //==========================================================================
    // 18. PRIVATE BATCH IMPLEMENTATION METHODS
    //==========================================================================

    /** @brief Scalar PDF kernel over a raw range, using a cache snapshot. */
    static void pdfKernel(const double* values, double* results, std::size_t count, double a,
                          double b, double d1, double d2, double log_pdf_const) noexcept;

    /** @brief Scalar log-PDF kernel over a raw range, using a cache snapshot. */
    static void logPdfKernel(const double* values, double* results, std::size_t count, double a,
                             double b, double d1, double d2, double log_pdf_const) noexcept;

    /** @brief Scalar CDF kernel over a raw range, using a cache snapshot. */
    static void cdfKernel(const double* values, double* results, std::size_t count, double a,
                          double b, double d1, double d2, double log_beta_prefix) noexcept;

    //==========================================================================
    // 19. PRIVATE COMPUTATIONAL METHODS
    //==========================================================================

    /** @brief Log-PDF from an explicit cache snapshot (shared by scalar/batch). */
    [[nodiscard]] static double logPdfImpl(double x, double a, double b, double d1, double d2,
                                           double log_pdf_const) noexcept;

    /** @brief CDF from an explicit cache snapshot (shared by scalar/batch). */
    [[nodiscard]] static double cdfImpl(double x, double a, double b, double d1, double d2,
                                        double log_beta_prefix) noexcept;

    /** @brief Survival from an explicit cache snapshot. */
    [[nodiscard]] static double sfImpl(double x, double a, double b, double d1, double d2,
                                       double log_beta_prefix) noexcept;

    /**
     * @brief Recompute the derived cache from d1_ / d2_ and resync `beta_`.
     * Called with a held unique_lock on cache_mutex_.
     */
    void updateCacheUnsafe() const noexcept override;

    //==========================================================================
    // 21. PRIVATE VALIDATION METHODS
    //==========================================================================

    /** @brief Validate F parameters, throwing on failure. */
    static void validateParameters(double d1, double d2) {
        auto v = ::stats::validateFisherFParameters(d1, d2);
        if (v.isError())
            throw std::invalid_argument(v.message());
    }

    //==========================================================================
    // 23. DISTRIBUTION PARAMETERS
    //==========================================================================

    /** @brief Numerator degrees of freedom d1 — positive and finite. */
    double d1_{detail::ONE};

    /** @brief Denominator degrees of freedom d2 — positive and finite. */
    double d2_{detail::ONE};

    /** @brief Atomic copies for lock-free parameter access. */
    mutable std::atomic<double> atomicD1_{detail::ONE};
    mutable std::atomic<double> atomicD2_{detail::ONE};
    mutable std::atomic<bool> atomicParamsValid_{false};

    //==========================================================================
    // 24. PERFORMANCE CACHE
    //==========================================================================

    /** @brief a = d1/2 — first incomplete-beta shape. */
    mutable double a_{detail::HALF};

    /** @brief b = d2/2 — second incomplete-beta shape. */
    mutable double b_{detail::HALF};

    /**
     * @brief a ln(d1) + b ln(d2) - ln B(a,b) — the constant part of the log-PDF.
     */
    mutable double logPdfConst_{detail::ZERO_DOUBLE};

    /**
     * @brief lgamma(a+b) - lgamma(a) - lgamma(b), the prefix accepted by
     * `detail::beta_i`'s four-argument overload. Symmetric in (a,b), so the
     * same value serves both the I_y(a,b) and the I_ybar(b,a) branch.
     */
    mutable double logBetaPrefix_{detail::ZERO_DOUBLE};

    //==========================================================================
    // 25. PRIVATE DELEGATION MEMBER
    //==========================================================================

    /**
     * @brief Internal Beta delegate — always maintained as Beta(d1/2, d2/2).
     *
     * Invariant: beta_.getAlpha() == d1_/2 and beta_.getBeta() == d2_/2.
     * Used for `sample()`; the probability functions deliberately bypass it
     * (see the class-level "Why not a pure delegation" note).
     */
    mutable BetaDistribution beta_{detail::HALF, detail::HALF};
};

}  // namespace stats

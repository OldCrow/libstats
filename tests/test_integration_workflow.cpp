/**
 * @file test_integration_workflow.cpp
 * @brief End-to-end integration test exercising the full libstats public API surface.
 *
 * TEST-2: validates that the major API layers interoperate correctly:
 *   - Factory construction via Result<T>
 *   - MLE fitting, statistical moments, support bounds
 *   - Survival and hazard functions
 *   - Scalar and batch probability / CDF / log-PDF
 *   - Goodness-of-fit (KS test via stats::analysis)
 *   - Bootstrap confidence intervals via stats::analysis
 *   - Cross-distribution consistency (NaN propagation, ±inf boundary values)
 *
 * This test intentionally uses multiple distributions in the same workflow
 * to catch API interoperability regressions rather than per-distribution
 * numerical accuracy (which is covered in the per-distribution test suites).
 */
#define LIBSTATS_FULL_INTERFACE
#include "libstats/libstats.h"
#include "libstats/stats/analysis/analysis.h"

#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <numeric>
#include <random>
#include <span>
#include <vector>

using namespace stats;

namespace {

/// Generate reproducible data from a given distribution.
template <typename Dist>
std::vector<double> generate_samples(Dist& dist, std::size_t n, std::uint32_t seed = 42) {
    std::mt19937 rng(seed);
    return dist.sample(rng, n);
}

constexpr double kTol = 1e-9;

}  // namespace

// ─────────────────────────────────────────────────────────────────────────────
// Factory construction and parameter round-trip
// ─────────────────────────────────────────────────────────────────────────────

TEST(IntegrationWorkflow, FactoryAndParameterRoundTrip) {
    // Gaussian
    auto g = GaussianDistribution::create(2.0, 0.5);
    ASSERT_TRUE(g.isOk());
    EXPECT_NEAR(g.value.getMean(), 2.0, kTol);
    EXPECT_NEAR(g.value.getVariance(), 0.25, kTol);

    // Exponential
    auto e = ExponentialDistribution::create(3.0);
    ASSERT_TRUE(e.isOk());
    EXPECT_NEAR(e.value.getMean(), 1.0 / 3.0, kTol);

    // Gamma
    auto gam = GammaDistribution::create(2.0, 1.0);
    ASSERT_TRUE(gam.isOk());
    EXPECT_NEAR(gam.value.getMean(), 2.0, kTol);
    EXPECT_NEAR(gam.value.getVariance(), 2.0, kTol);

    // Beta
    auto b = BetaDistribution::create(2.0, 3.0);
    ASSERT_TRUE(b.isOk());
    EXPECT_NEAR(b.value.getMean(), 2.0 / 5.0, 1e-8);

    // Invalid parameters are rejected
    EXPECT_TRUE(GaussianDistribution::create(0.0, -1.0).isError());
    EXPECT_TRUE(ExponentialDistribution::create(-1.0).isError());
    EXPECT_TRUE(GammaDistribution::create(0.0, 1.0).isError());
}

// ─────────────────────────────────────────────────────────────────────────────
// MLE fitting: fitted distribution should recover approximate parameters
// ─────────────────────────────────────────────────────────────────────────────

TEST(IntegrationWorkflow, MLEFitting) {
    std::mt19937 rng(123);

    // Gaussian: fit to N(5, 2) samples
    {
        auto true_dist = GaussianDistribution::create(5.0, 2.0).value;
        auto data = generate_samples(true_dist, 2000, 123);
        GaussianDistribution fitted;
        ASSERT_NO_THROW(fitted.fit(data));
        EXPECT_NEAR(fitted.getMean(), 5.0, 0.2);
        EXPECT_NEAR(fitted.getVariance(), 4.0, 0.5);
    }

    // Exponential: fit to Exp(2) samples
    {
        auto true_dist = ExponentialDistribution::create(2.0).value;
        auto data = generate_samples(true_dist, 2000, 456);
        ExponentialDistribution fitted;
        ASSERT_NO_THROW(fitted.fit(data));
        EXPECT_NEAR(fitted.getMean(), 0.5, 0.05);
    }

    // Poisson: fit to Poisson(4) samples
    {
        auto true_dist = PoissonDistribution::create(4.0).value;
        auto data = generate_samples(true_dist, 2000, 789);
        PoissonDistribution fitted;
        ASSERT_NO_THROW(fitted.fit(data));
        EXPECT_NEAR(fitted.getMean(), 4.0, 0.3);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Survival and hazard functions
// ─────────────────────────────────────────────────────────────────────────────

TEST(IntegrationWorkflow, SurvivalAndHazardFunctions) {
    auto exp_dist = ExponentialDistribution::create(1.0).value;
    auto weibull_dist = WeibullDistribution::create(2.0, 1.0).value;

    // Exponential: S(x) = exp(-lambda*x), H(x) = lambda (constant)
    for (double x : {0.5, 1.0, 2.0}) {
        double s = exp_dist.getSurvival(x);
        double f = exp_dist.getCumulativeProbability(x);
        EXPECT_NEAR(s + f, 1.0, 1e-10) << "S(x) + F(x) = 1 violated at x=" << x;
        EXPECT_GT(s, 0.0);
        EXPECT_LT(s, 1.0);

        double h = exp_dist.getHazard(x);
        EXPECT_NEAR(h, 1.0, 1e-6) << "Exponential hazard should be constant (lambda=1)";
    }

    // Weibull: S(0) = 1, S(+inf) approaches 0
    EXPECT_NEAR(weibull_dist.getSurvival(0.0), 1.0, 1e-10);
    EXPECT_NEAR(weibull_dist.getSurvival(1e6), 0.0, 1e-6);

    // Boundary: survival of -inf should be 1 (no mass below -inf)
    EXPECT_NEAR(exp_dist.getSurvival(-1e300), 1.0, 1e-10);
}

// ─────────────────────────────────────────────────────────────────────────────
// Batch probability and CDF match scalar results
// ─────────────────────────────────────────────────────────────────────────────

TEST(IntegrationWorkflow, BatchMatchesScalar) {
    auto dist = GaussianDistribution::create(0.0, 1.0).value;

    const std::size_t N = 100;
    std::vector<double> xs(N), pdf_batch(N), cdf_batch(N), logpdf_batch(N);
    for (std::size_t i = 0; i < N; ++i)
        xs[i] = -3.0 + 6.0 * static_cast<double>(i) / static_cast<double>(N - 1);

    dist.getProbability(std::span<const double>(xs), std::span<double>(pdf_batch));
    dist.getCumulativeProbability(std::span<const double>(xs), std::span<double>(cdf_batch));
    dist.getLogProbability(std::span<const double>(xs), std::span<double>(logpdf_batch));

    for (std::size_t i = 0; i < N; ++i) {
        EXPECT_NEAR(pdf_batch[i], dist.getProbability(xs[i]), 1e-10)
            << "Batch PDF mismatch at i=" << i;
        EXPECT_NEAR(cdf_batch[i], dist.getCumulativeProbability(xs[i]), 1e-10)
            << "Batch CDF mismatch at i=" << i;
        EXPECT_NEAR(logpdf_batch[i], dist.getLogProbability(xs[i]), 1e-10)
            << "Batch log-PDF mismatch at i=" << i;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// NaN propagation convention: all distributions return NaN for NaN input
// ─────────────────────────────────────────────────────────────────────────────

TEST(IntegrationWorkflow, NaNInputPropagatesNaNOutput) {
    const double nan = std::numeric_limits<double>::quiet_NaN();

    auto check = [&](const char* name, auto& dist) {
        EXPECT_TRUE(std::isnan(dist.getProbability(nan)))
            << name << "::getProbability(NaN) should return NaN";
        EXPECT_TRUE(std::isnan(dist.getCumulativeProbability(nan)))
            << name << "::getCumulativeProbability(NaN) should return NaN";
        EXPECT_TRUE(std::isnan(dist.getLogProbability(nan)))
            << name << "::getLogProbability(NaN) should return NaN";
    };

    auto gaussian     = GaussianDistribution::create(0.0, 1.0).value;
    auto exponential  = ExponentialDistribution::create(1.0).value;
    auto uniform      = UniformDistribution::create(0.0, 1.0).value;
    auto gamma        = GammaDistribution::create(2.0, 1.0).value;
    auto chisquared   = ChiSquaredDistribution::create(3.0).value;
    auto student      = StudentTDistribution::create(5.0).value;
    auto beta         = BetaDistribution::create(2.0, 2.0).value;
    auto lognormal    = LogNormalDistribution::create(0.0, 1.0).value;
    auto pareto       = ParetoDistribution::create(1.0, 2.0).value;
    auto weibull      = WeibullDistribution::create(2.0, 1.0).value;
    auto rayleigh     = RayleighDistribution::create(1.0).value;
    auto vonmises     = VonMisesDistribution::create(0.0, 1.0).value;
    auto binomial     = BinomialDistribution::create(10, 0.5).value;
    auto negbinomial  = NegativeBinomialDistribution::create(3.0, 0.5).value;
    auto geometric    = GeometricDistribution::create(0.5).value;
    auto poisson      = PoissonDistribution::create(3.0).value;
    auto discrete     = DiscreteDistribution::create(0, 9).value;

    check("Gaussian",         gaussian);
    check("Exponential",      exponential);
    check("Uniform",          uniform);
    check("Gamma",            gamma);
    check("ChiSquared",       chisquared);
    check("StudentT",         student);
    check("Beta",             beta);
    check("LogNormal",        lognormal);
    check("Pareto",           pareto);
    check("Weibull",          weibull);
    check("Rayleigh",         rayleigh);
    check("VonMises",         vonmises);
    check("Binomial",         binomial);
    check("NegativeBinomial", negbinomial);
    check("Geometric",        geometric);
    check("Poisson",          poisson);
    check("Discrete",         discrete);
}

// ────────────────────────────────────────────────────────────────────────────────
// Geometric end-to-end workflow
// ────────────────────────────────────────────────────────────────────────────────

TEST(IntegrationWorkflow, GeometricWorkflow) {
    // Convention: X = failures before first success, support {0,1,2,...}
    // Delegation: wraps NegativeBinomial(r=1, p)
    auto r = GeometricDistribution::create(0.4);
    ASSERT_TRUE(r.isOk());
    auto g = std::move(r.value);

    // Moment formulas
    EXPECT_NEAR(g.getMean(),     0.6 / 0.4,          1e-10);
    EXPECT_NEAR(g.getVariance(), 0.6 / (0.4 * 0.4),  1e-10);
    EXPECT_EQ(g.getMode(), 0.0);
    EXPECT_TRUE(g.isDiscrete());

    // Known PMF: PMF(0) = p = 0.4, PMF(1) = p*(1-p) = 0.24
    EXPECT_NEAR(g.getProbability(0.0), 0.4,  1e-12);
    EXPECT_NEAR(g.getProbability(1.0), 0.24, 1e-12);
    EXPECT_EQ(g.getProbability(-1.0), 0.0);  // out of support

    // Known CDF: CDF(0) = p = 0.4
    EXPECT_NEAR(g.getCumulativeProbability(0.0), 0.4, 1e-12);

    // Sampling: all values >= 0
    std::mt19937 rng(42);
    auto samples = g.sample(rng, 200);
    EXPECT_EQ(samples.size(), 200u);
    for (double v : samples)
        EXPECT_GE(v, 0.0) << "Geometric samples must be non-negative";

    // MLE round-trip: fit to Geometric(0.4) data
    g.fit(samples);
    EXPECT_NEAR(g.getP(), 0.4, 0.08) << "MLE p should recover ~0.4";

    // Invalid parameters rejected
    EXPECT_TRUE(GeometricDistribution::create(0.0).isError());
    EXPECT_TRUE(GeometricDistribution::create(-1.0).isError());
    EXPECT_TRUE(GeometricDistribution::create(1.5).isError());
}

// ─────────────────────────────────────────────────────────────────────────────
// CDF boundary values: F(-inf) = 0, F(+inf) = 1
// ─────────────────────────────────────────────────────────────────────────────

TEST(IntegrationWorkflow, CDFBoundaryValues) {
    const double ninf = -std::numeric_limits<double>::infinity();
    const double pinf =  std::numeric_limits<double>::infinity();

    auto check_cdf = [&](const char* name, auto& dist) {
        EXPECT_NEAR(dist.getCumulativeProbability(ninf), 0.0, 1e-10)
            << name << "::CDF(-inf) should be 0";
        EXPECT_NEAR(dist.getCumulativeProbability(pinf), 1.0, 1e-10)
            << name << "::CDF(+inf) should be 1";
    };

    auto gaussian  = GaussianDistribution::create(0.0, 1.0).value;
    auto gamma     = GammaDistribution::create(2.0, 1.0).value;
    auto student   = StudentTDistribution::create(5.0).value;
    auto weibull   = WeibullDistribution::create(2.0, 1.0).value;
    auto pareto    = ParetoDistribution::create(1.0, 2.0).value;
    auto exponential = ExponentialDistribution::create(1.0).value;

    check_cdf("Gaussian",     gaussian);
    check_cdf("Gamma",        gamma);
    check_cdf("StudentT",     student);
    check_cdf("Weibull",      weibull);
    check_cdf("Pareto",       pareto);
    check_cdf("Exponential",  exponential);
}

// ─────────────────────────────────────────────────────────────────────────────
// Goodness-of-fit: KS test on samples drawn from the tested distribution
// should not reject at a tight significance level
// ─────────────────────────────────────────────────────────────────────────────

TEST(IntegrationWorkflow, KSTestOnOwnSamples) {
    auto dist = GaussianDistribution::create(0.0, 1.0).value;
    auto data = generate_samples(dist, 500, 999);

    // KS test: samples from N(0,1) vs N(0,1) should not reject H0
    auto [ks_stat, p_value, rejected] =
        stats::analysis::kolmogorovSmirnovTest(data, dist, 0.01);

    EXPECT_FALSE(rejected)
        << "KS test falsely rejected H0 for own samples (p=" << p_value << ")";
    EXPECT_GT(p_value, 0.0);
    EXPECT_LT(ks_stat, 0.2)
        << "KS statistic unexpectedly large for own samples: " << ks_stat;
}

// ─────────────────────────────────────────────────────────────────────────────
// Bootstrap confidence interval: CI should contain the true parameter
// ─────────────────────────────────────────────────────────────────────────────

TEST(IntegrationWorkflow, BootstrapCIContainsTrueParameter) {
    // Generate data from Exponential(2) — true mean = 0.5
    auto true_dist = ExponentialDistribution::create(2.0).value;
    auto data = generate_samples(true_dist, 2000, 777);  // large n for stability

    ExponentialDistribution fitted;
    fitted.fit(data);

    auto [lo, hi] = stats::analysis::bootstrapMeanCI<ExponentialDistribution>(
        data, 0.95, 500);

    // The 95% CI should be non-degenerate and straddling the true mean (0.5).
    // With n=2000 the sample mean will be very close to 0.5, so the CI
    // will comfortably bracket it.  We also verify the CI is sane.
    const double true_mean = 0.5;
    EXPECT_LT(lo, hi)  << "Bootstrap CI is degenerate (lo >= hi)";
    EXPECT_LT(lo, true_mean)
        << "Bootstrap CI lower bound (" << lo << ") exceeds true mean (" << true_mean << ")";
    EXPECT_GT(hi, true_mean)
        << "Bootstrap CI upper bound (" << hi << ") is below true mean (" << true_mean << ")";
}

// ─────────────────────────────────────────────────────────────────────────────
// reset() followed by fit() should work without leftover state
// ─────────────────────────────────────────────────────────────────────────────

TEST(IntegrationWorkflow, ResetAndRefit) {
    GaussianDistribution dist;
    // Default parameters: mean=0, stddev=1

    std::mt19937 rng(42);
    // Fit to N(10, 3) data
    auto target = GaussianDistribution::create(10.0, 3.0).value;
    auto data = generate_samples(target, 1000, 42);
    dist.fit(data);
    EXPECT_NEAR(dist.getMean(), 10.0, 0.5);

    // Reset then fit to N(-5, 0.5) data
    dist.reset();
    EXPECT_NEAR(dist.getMean(), 0.0, kTol);

    auto target2 = GaussianDistribution::create(-5.0, 0.5).value;
    auto data2 = generate_samples(target2, 1000, 99);
    dist.fit(data2);
    EXPECT_NEAR(dist.getMean(), -5.0, 0.2);
    EXPECT_NEAR(std::sqrt(dist.getVariance()), 0.5, 0.1);
}

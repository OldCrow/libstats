/**
 * @file distribution_families_demo.cpp
 * @brief libstats distributions organized by statistical family
 *
 * Distributions are grouped into four families based on what they model.
 * For each family this example explains:
 *   - what the family is for (when should you reach for it at all?)
 *   - the distinguishing properties of each member
 *   - a concrete scenario that motivates the choice
 *   - a within-family comparison showing when the members diverge
 *
 * Reading order: work through the families top to bottom. Each section is
 * self-contained — you can also jump directly to the family you need.
 */

#define LIBSTATS_FULL_INTERFACE
#include "libstats/libstats.h"

#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

namespace {

void section(const std::string& title) {
    std::cout << "\n"
              << std::string(72, '=') << "\n"
              << title << "\n"
              << std::string(72, '=') << "\n";
}

void subsection(const std::string& title) {
    std::cout << "\n--- " << title << " ---\n";
}

}  // namespace

// ==========================================================================
// FAMILY 1: Symmetric full-domain continuous
// ==========================================================================
//
// Use these distributions when the quantity can be any real number and
// natural variation is approximately symmetric around a central value.
// Examples: measurement error, test scores, log-returns in finance,
// residuals from a regression model.
//
// The two members differ in how much weight they put in the tails:
//   Gaussian    -- the baseline; use it when you have a large sample or
//                  when the Central Limit Theorem applies.
//   Student's t -- use it when your sample is small (n < ~30) or when you
//                  want heavier tails than Gaussian allows.
// ==========================================================================

void demo_symmetric_continuous() {
    section("Family 1: Symmetric full-domain continuous");

    std::cout << "\n"
              << "These distributions model real-valued quantities that are naturally\n"
              << "centered and symmetric. They arise whenever many small independent\n"
              << "influences add up -- the Central Limit Theorem explains why the\n"
              << "Gaussian appears so widely.\n";

    // --- Gaussian -----------------------------------------------------------
    subsection("Gaussian (Normal) distribution");
    std::cout << "\n"
              << "Parameters: mean mu (location), std dev sigma (spread).\n"
              << "Use when: sample size is large enough that the CLT applies, or\n"
              << "          the data genuinely comes from a Gaussian process.\n"
              << "\n"
              << "Scenario: A production process fills 500 mL bottles. Repeated\n"
              << "measurements show fills are N(500, 1.5). What fraction of bottles\n"
              << "are underfilled (< 497 mL)?\n";
    auto fill = stats::GaussianDistribution::create(500.0, 1.5).value;
    double p_under = fill.getCumulativeProbability(497.0);
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  P(fill < 497 mL)   = " << p_under << "  (~" << p_under * 100
              << "% of bottles)\n";
    std::cout << "  99th percentile    = " << fill.getQuantile(0.99)
              << " mL  (SPC upper control limit)\n";

    // --- Student's t --------------------------------------------------------
    subsection("Student's t distribution");
    std::cout << "\n"
              << "Parameters: degrees of freedom nu > 0. Location=0, scale=1.\n"
              << "  For nu=1 this is the Cauchy distribution (undefined mean).\n"
              << "  As nu -> inf the distribution converges to Gaussian.\n"
              << "Use when: sample is small and true variance is unknown, or when\n"
              << "          heavier tails than Gaussian are needed.\n"
              << "\n"
              << "Scenario: You run an A/B test with only 8 observations per group.\n"
              << "The critical value for a one-sample t-test at 5% significance\n"
              << "(two-tailed, df=7) is:\n";
    auto t7 = stats::StudentTDistribution::create(7.0).value;
    auto z = stats::GaussianDistribution::create(0.0, 1.0).value;
    double t_crit = t7.getQuantile(0.975);
    double z_crit = z.getQuantile(0.975);
    std::cout << "  t_{0.975, df=7}  = " << t_crit << "\n";
    std::cout << "  z_{0.975}        = " << z_crit << "  (Gaussian; used when n is large)\n";

    // --- Within-family comparison -------------------------------------------
    subsection("When does Student's t differ meaningfully from Gaussian?");
    std::cout << "\n"
              << "The practical difference is in the tails. For large nu the two are\n"
              << "nearly identical; for small nu the t-distribution puts substantially\n"
              << "more probability far from zero.\n"
              << "\n"
              << "P(|X| > 2.5) for different degrees of freedom vs. Gaussian:\n";
    double p_gauss = 2.0 * (1.0 - z.getCumulativeProbability(2.5));
    std::cout << "  Gaussian (nu->inf):  " << p_gauss * 100 << "%\n";
    for (double nu : {2.0, 5.0, 15.0, 30.0}) {
        auto t = stats::StudentTDistribution::create(nu).value;
        double p = 2.0 * (1.0 - t.getCumulativeProbability(2.5));
        std::cout << "  Student's t(nu=" << std::setw(2) << (int)nu << "):  " << p * 100 << "%\n";
    }
    std::cout << "\nRule of thumb: use Student's t when n < 30 and sigma is\n"
              << "estimated from data.\n";
}

// ==========================================================================
// FAMILY 2: Positive-support continuous
// ==========================================================================
//
// Use these distributions when the quantity is necessarily positive.
// Common examples: waiting times, lifetimes, physical measurements
// (mass, length, intensity), variance estimates from normal populations.
//
//   Exponential -- single parameter; memoryless; waiting time for one
//                  Poisson-process event.
//   Gamma       -- two parameters; generalizes Exponential; waiting time
//                  for k events, or any flexible right-skewed positive qty.
//   Chi-squared -- one-parameter special case of Gamma; arises from sums
//                  of squared standard normals; used in hypothesis tests.
// ==========================================================================

void demo_positive_support() {
    section("Family 2: Positive-support continuous");

    std::cout << "\n"
              << "These distributions model quantities that cannot be negative --\n"
              << "waiting times, durations, physical measurements. The key question:\n"
              << "how much shape flexibility do you need?\n";

    // --- Exponential --------------------------------------------------------
    subsection("Exponential distribution");
    std::cout << "\n"
              << "Parameters: rate lambda > 0.  Mean = 1/lambda.\n"
              << "Use when: modelling time until the next event in a Poisson process\n"
              << "          (requests, failures, decays).\n"
              << "Key property: memoryless -- the remaining wait has the same\n"
              << "              distribution regardless of elapsed time.\n"
              << "\n"
              << "Scenario: A web server receives requests at 20/second (lambda=20).\n"
              << "What is the probability that the next request arrives within 0.1 s?\n";
    auto req = stats::ExponentialDistribution::create(20.0).value;
    std::cout << "  Mean inter-arrival   = " << req.getMean() * 1000 << " ms\n";
    std::cout << "  P(next <= 0.1 s)     = " << req.getCumulativeProbability(0.1) << "\n";
    std::cout << "  P(wait > 0.2 s)      = " << (1.0 - req.getCumulativeProbability(0.2)) << "\n";

    // --- Gamma --------------------------------------------------------------
    subsection("Gamma distribution");
    std::cout << "\n"
              << "Parameters: shape alpha > 0, rate beta > 0.  Mean = alpha/beta.\n"
              << "  Gamma(alpha=1, beta) is identical to Exponential(beta).\n"
              << "  Gamma(alpha=k, beta) is the waiting time for k independent\n"
              << "  Exponential(beta) events (e.g. k sequential pipeline stages).\n"
              << "Use when: wait time is not memoryless, or you need a flexible\n"
              << "          unimodal right-skewed distribution.\n"
              << "\n"
              << "Scenario: A support ticket needs sign-off from 3 reviewers, each\n"
              << "taking Exponential(0.5 hr^-1) time. Total time ~ Gamma(3, 0.5).\n"
              << "What is the 90th percentile completion time?\n";
    auto review = stats::GammaDistribution::create(3.0, 0.5).value;
    std::cout << std::setprecision(2);
    std::cout << "  Mean completion       = " << review.getMean() << " hours\n";
    std::cout << "  90th percentile       = " << review.getQuantile(0.90) << " hours\n";
    std::cout << std::setprecision(4);
    std::cout << "  P(done within 8 hrs)  = " << review.getCumulativeProbability(8.0) << "\n";

    // --- Chi-squared --------------------------------------------------------
    subsection("Chi-squared distribution");
    std::cout << "\n"
              << "Parameter: degrees of freedom nu > 0.\n"
              << "  chi^2(nu) = Gamma(nu/2, 1/2) exactly.\n"
              << "  It is the distribution of the sum of nu squared standard normals.\n"
              << "Use when: working with variance estimates or test statistics in\n"
              << "          chi-squared tests, F-tests, or likelihood ratio tests.\n"
              << "\n"
              << "Scenario: You test whether a population variance equals sigma^2=4\n"
              << "using n=10 observations. The statistic (n-1)*S^2/sigma^2 follows\n"
              << "chi^2(df=9) under H0. The two-tailed critical region at alpha=0.05:\n";
    auto chi2_9 = stats::ChiSquaredDistribution::create(9.0).value;
    std::cout << std::setprecision(3);
    std::cout << "  Lower critical value  = " << chi2_9.getQuantile(0.025) << "\n";
    std::cout << "  Upper critical value  = " << chi2_9.getQuantile(0.975) << "\n";
    std::cout << "  Mean = " << chi2_9.getMean() << "  (always equals nu)\n";

    // --- Within-family comparison -------------------------------------------
    subsection("Choosing within the positive-support family");
    std::cout << "\n"
              << "All three share (0, inf) support. Choose by asking:\n"
              << "\n"
              << "  Waiting for one event at constant rate?          -> Exponential\n"
              << "  Waiting for k events, or flexible skewed shape?  -> Gamma\n"
              << "  Sum of squared normals, variance test statistic? -> Chi-squared\n"
              << "\n"
              << "Note: Chi-squared IS Gamma. The chi-squared parameterisation\n"
              << "matches standard statistical tables directly.\n"
              << "\n"
              << "Verification -- Gamma(alpha=1, beta=2) and Exponential(lambda=2)\n"
              << "are the same distribution:\n";
    auto gamma_1_2 = stats::GammaDistribution::create(1.0, 2.0).value;
    auto expo_2 = stats::ExponentialDistribution::create(2.0).value;
    std::cout << std::setprecision(6);
    std::cout << "  Gamma(1,2) PDF(0.5)  = " << gamma_1_2.getProbability(0.5) << "\n";
    std::cout << "  Expo(2)    PDF(0.5)  = " << expo_2.getProbability(0.5) << "  (identical)\n";
}

// ==========================================================================
// FAMILY 3: Bounded continuous
// ==========================================================================
//
// Use these distributions when the quantity is constrained to a finite
// interval -- a proportion, a probability, a bounded physical measurement.
//
//   Uniform -- maximum entropy on [a, b]; use when every value in the
//              range is equally plausible and you have no other information.
//   Beta    -- flexible shape on [0, 1]; use when you have information
//              about a proportion or probability (Bayesian prior, click-
//              through rates, success fractions, mixture weights).
// ==========================================================================

void demo_bounded_continuous() {
    section("Family 3: Bounded continuous");

    std::cout << "\n"
              << "These distributions model quantities restricted to a finite interval.\n"
              << "Key question: do you have shape information, or are all values\n"
              << "equally plausible?\n";

    // --- Uniform ------------------------------------------------------------
    subsection("Uniform distribution");
    std::cout << "\n"
              << "Parameters: lower bound a, upper bound b.  Mean = (a+b)/2.\n"
              << "Use when: any value in [a, b] is equally likely -- you have no\n"
              << "          reason to prefer one sub-range over another. Common in\n"
              << "          simulation, random tie-breaking, initial guesses.\n"
              << "\n"
              << "Scenario: A project task is estimated to take between 3 and 7 days\n"
              << "with no further information about the shape.\n";
    auto task = stats::UniformDistribution::create(3.0, 7.0).value;
    std::cout << std::setprecision(4);
    std::cout << "  Expected duration      = " << task.getMean() << " days\n";
    std::cout << "  P(done within 5 days)  = " << task.getCumulativeProbability(5.0) << "\n";
    std::cout << "  Std deviation          = " << stats::getStandardDeviation(task) << " days\n";

    // --- Beta ---------------------------------------------------------------
    subsection("Beta distribution");
    std::cout << "\n"
              << "Parameters: shape alpha > 0, shape beta > 0.  Support [0,1].\n"
              << "  Beta(1, 1) = Uniform(0, 1).\n"
              << "  alpha > beta -> skewed toward 1 (high success rates more likely).\n"
              << "  alpha = beta -> symmetric; larger values concentrate around 0.5.\n"
              << "Use when: modelling a proportion or probability, especially with\n"
              << "          prior information. Bayesian update rule: observe k\n"
              << "          successes in n trials -> posterior Beta(alpha+k, beta+n-k).\n"
              << "\n"
              << "Scenario: A new landing page has shown 6 conversions in 20 views.\n"
              << "Starting from a uniform prior Beta(1,1), the posterior on the\n"
              << "conversion rate is Beta(7, 15). What is the 95% credible interval?\n";
    auto posterior = stats::BetaDistribution::create(7.0, 15.0).value;
    std::cout << "  Posterior mean         = " << std::setprecision(4) << posterior.getMean()
              << "  (point estimate)\n";
    std::cout << "  95% credible interval  = [" << posterior.getQuantile(0.025) << ", "
              << posterior.getQuantile(0.975) << "]\n";
    std::cout << "  P(true rate > 0.5)     = " << (1.0 - posterior.getCumulativeProbability(0.5))
              << "\n";

    // --- Within-family comparison -------------------------------------------
    subsection("Uniform vs. Beta: the role of prior information");
    std::cout << "\n"
              << "Both have [0,1] support but very different shapes.\n"
              << "Beta(1,1) IS Uniform -- verifying this is a useful sanity check:\n";
    auto beta_1_1 = stats::BetaDistribution::create(1.0, 1.0).value;
    auto uniform_01 = stats::UniformDistribution::create(0.0, 1.0).value;
    std::cout << "  Beta(1,1) PDF(0.4)     = " << beta_1_1.getProbability(0.4) << "\n";
    std::cout << "  Uniform(0,1) PDF(0.4)  = " << uniform_01.getProbability(0.4) << "  (same)\n";
    std::cout << "\n"
              << "When you have prior knowledge or observed data, use Beta over\n"
              << "Uniform: the posterior mean is shrunk toward the prior, reducing\n"
              << "variance in small samples.\n";
}

// ==========================================================================
// FAMILY 4: Discrete
// ==========================================================================
//
// Use these distributions when the outcome is a count or integer -- you
// cannot observe 2.7 events or 1.4 items.
//
//   Poisson  -- number of independent events in a fixed time or space
//               when the rate is constant. Mean equals variance.
//   Discrete -- uniform draw over a finite integer range. Each integer
//               value is equally likely.
// ==========================================================================

void demo_discrete() {
    section("Family 4: Discrete distributions");

    std::cout << "\n"
              << "These distributions model counts and categories. The quantity is\n"
              << "always an integer; fractional values have zero probability.\n";

    // --- Poisson ------------------------------------------------------------
    subsection("Poisson distribution");
    std::cout << "\n"
              << "Parameter: rate lambda > 0.  Mean = Variance = lambda.\n"
              << "Use when: counting independent events at a constant average rate\n"
              << "          over a fixed interval (server requests, defects, arrivals).\n"
              << "Key property: mean equals variance. If sample variance >> sample\n"
              << "              mean, consider an overdispersed model instead.\n"
              << "\n"
              << "Scenario: A call centre receives 12 calls/hour on average.\n"
              << "In a 30-minute window (lambda=6), what is the probability of\n"
              << "10 or more calls (triggering an alert threshold)?\n";
    auto calls = stats::PoissonDistribution::create(6.0).value;
    double p_over = 1.0 - calls.getCumulativeProbability(9);
    std::cout << std::setprecision(4);
    std::cout << "  lambda = 6 calls / 30 min\n";
    std::cout << "  P(calls >= 10)   = " << p_over << "\n";
    std::cout << "  P(calls = 6)     = " << calls.getProbability(6) << "  (PMF at mode)\n";
    std::cout << "  90th percentile  = " << (int)calls.getQuantile(0.90) << " calls\n";

    // --- Discrete uniform ---------------------------------------------------
    subsection("Discrete uniform distribution");
    std::cout << "\n"
              << "Parameters: integer bounds [min, max].  Mean = (min+max)/2.\n"
              << "Use when: every integer in the range is equally likely -- a fair\n"
              << "          die roll, a random selection from a numbered list.\n"
              << "\n"
              << "Scenario: Fair six-sided die -- expected value and P(roll >= 5):\n";
    auto die = stats::DiscreteDistribution::create(1, 6).value;
    std::cout << "  Expected value    = " << die.getMean() << "\n";
    std::cout << "  P(roll >= 5)      = " << (1.0 - die.getCumulativeProbability(4)) << "\n";
    std::cout << "  P(roll = 3)       = " << die.getProbability(3) << "  (1/6 for all faces)\n";
}

// ==========================================================================
// BATCH API: the same interface across all distributions
// ==========================================================================
//
// All distributions share a span-based batch API. For large arrays this is
// significantly faster than calling getProbability() in a loop because the
// library selects SIMD and parallel strategies automatically.
// ==========================================================================

void demo_batch_api() {
    section("Batch API: consistent across all families");

    std::cout << "\n"
              << "Every distribution supports the same batch interface:\n"
              << "  dist.getProbability(std::span<const double>, std::span<double>)\n"
              << "  dist.getCumulativeProbability(...)\n"
              << "  dist.getLogProbability(...)\n"
              << "\n"
              << "The library selects scalar, SIMD-vectorised, or parallel execution\n"
              << "based on array size and machine capabilities. Nothing to configure.\n";

    std::vector<double> xs = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0};
    std::vector<double> out(xs.size());

    auto gamma = stats::GammaDistribution::create(2.0, 1.0).value;
    gamma.getProbability(xs, out);

    std::cout << "\nGamma(alpha=2, beta=1) PDF across a grid:\n";
    std::cout << std::fixed << std::setprecision(4);
    for (size_t i = 0; i < xs.size(); ++i) {
        std::cout << "  x=" << xs[i] << "  ->  " << out[i] << "\n";
    }
    std::cout << "\nThe same call works identically for all 9 distributions.\n";
}

int main() {
    stats::initialize_performance_systems();

    demo_symmetric_continuous();
    demo_positive_support();
    demo_bounded_continuous();
    demo_discrete();
    demo_batch_api();

    std::cout << "\n" << std::string(72, '=') << "\n";
    std::cout << "9 distributions across 4 families.\n";
    std::cout << "See the individual distribution headers for the full API surface.\n";
    std::cout << std::string(72, '=') << "\n";

    return 0;
}

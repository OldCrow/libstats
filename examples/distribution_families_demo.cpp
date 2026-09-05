/**
 * @file distribution_families_demo.cpp
 * @brief libstats distributions organized by statistical family
 *
 * All 27 distributions are grouped into seven families based on what they
 * model. For each family this example explains:
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
// The members differ in how much weight they put in the tails:
//   Gaussian    -- the baseline; use it when you have a large sample or
//                  when the Central Limit Theorem applies.
//   Student's t -- use it when your sample is small (n < ~30) or when you
//                  want heavier tails than Gaussian allows.
//   Logistic    -- slightly heavier tails than Gaussian; its CDF is the
//                  closed-form sigmoid, so it underlies logistic regression.
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
    auto fill = stats::GaussianDistribution::create(500.0, 1.5).unwrap();
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
    auto t7 = stats::StudentTDistribution::create(7.0).unwrap();
    auto z = stats::GaussianDistribution::create(0.0, 1.0).unwrap();
    double t_crit = t7.getQuantile(0.975);
    double z_crit = z.getQuantile(0.975);
    std::cout << "  t_{0.975, df=7}  = " << t_crit << "\n";
    std::cout << "  z_{0.975}        = " << z_crit << "  (Gaussian; used when n is large)\n";

    // --- Logistic -----------------------------------------------------------
    subsection("Logistic distribution");
    std::cout << "\n"
              << "Parameters: location mu, scale s > 0.  Mean = mu.\n"
              << "  The CDF has the closed form 1/(1 + exp(-(x-mu)/s)) -- the\n"
              << "  sigmoid. Tails are heavier than Gaussian (kurtosis 4.2 vs 3).\n"
              << "Use when: modelling the latent noise in a binary-choice process\n"
              << "          (logistic regression assumes exactly this), or growth\n"
              << "          curves, or when you want a Gaussian-like shape with a\n"
              << "          closed-form CDF and quantile.\n"
              << "\n"
              << "Scenario: A pass/fail exam is modelled with a latent skill score\n"
              << "~ Logistic(mu=60, s=8). The pass mark is 70. What fraction pass?\n";
    auto skill = stats::LogisticDistribution::create(60.0, 8.0).unwrap();
    std::cout << std::setprecision(4);
    std::cout << "  P(score >= 70)     = " << (1.0 - skill.getCumulativeProbability(70.0)) << "\n";
    std::cout << "  Median             = " << skill.getQuantile(0.5) << "  (= mu, symmetric)\n";
    std::cout << "  90th percentile    = " << skill.getQuantile(0.90) << "\n";

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
        auto t = stats::StudentTDistribution::create(nu).unwrap();
        double p = 2.0 * (1.0 - t.getCumulativeProbability(2.5));
        std::cout << "  Student's t(nu=" << std::setw(2) << static_cast<int>(nu)
                  << "):  " << p * 100 << "%\n";
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
    auto req = stats::ExponentialDistribution::create(20.0).unwrap();
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
    auto review = stats::GammaDistribution::create(3.0, 0.5).unwrap();
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
    auto chi2_9 = stats::ChiSquaredDistribution::create(9.0).unwrap();
    std::cout << std::setprecision(3);
    std::cout << "  Lower critical value  = " << chi2_9.getQuantile(0.025) << "\n";
    std::cout << "  Upper critical value  = " << chi2_9.getQuantile(0.975) << "\n";
    std::cout << "  Mean = " << chi2_9.getMean() << "  (always equals nu)\n";

    // --- LogNormal ----------------------------------------------------------
    subsection("Log-Normal distribution");
    std::cout << "\n"
              << "Parameters: mu (log-scale location), sigma (log-scale spread > 0).\n"
              << "  The logarithm of a LogNormal variable is Gaussian(mu, sigma).\n"
              << "  Median = exp(mu); Mean = exp(mu + sigma^2/2).\n"
              << "Use when: a multiplicative process generates the data (income,\n"
              << "          file sizes, latency, reaction times).\n"
              << "\n"
              << "Scenario: Service latency is log-normally distributed.\n"
              << "LogNormal(mu=5, sigma=0.5) has median exp(5) = 148 ms.\n"
              << "What fraction of requests take longer than 300 ms?\n";
    auto latency = stats::LogNormalDistribution::create(5.0, 0.5).unwrap();
    std::cout << std::setprecision(4);
    std::cout << "  Median              = " << latency.getMedian() << " ms\n";
    std::cout << "  Mean                = " << latency.getMean() << " ms\n";
    std::cout << "  P(latency > 300ms)  = " << (1.0 - latency.getCumulativeProbability(300.0))
              << "\n";

    // --- Weibull ------------------------------------------------------------
    subsection("Weibull distribution");
    std::cout << "\n"
              << "Parameters: shape k > 0, scale lambda > 0.\n"
              << "  k < 1 -> decreasing hazard (infant mortality)\n"
              << "  k = 1 -> constant hazard, identical to Exponential(1/lambda)\n"
              << "  k > 1 -> increasing hazard (wear-out failures)\n"
              << "Use when: modelling component lifetimes in reliability engineering,\n"
              << "          or any quantity with a time-varying failure rate.\n"
              << "\n"
              << "Scenario: A motor bearing has a Weibull(k=2.5, lambda=1000 hr)\n"
              << "lifetime. What fraction survive beyond 800 hours?\n";
    auto bearing = stats::WeibullDistribution::create(2.5, 1000.0).unwrap();
    std::cout << std::setprecision(4);
    std::cout << "  Mean lifetime       = " << bearing.getMean() << " hr\n";
    std::cout << "  P(survive > 800 hr) = " << (1.0 - bearing.getCumulativeProbability(800.0))
              << "\n";
    std::cout << "  10th percentile     = " << bearing.getQuantile(0.10) << " hr\n";

    // --- Pareto -------------------------------------------------------------
    subsection("Pareto distribution");
    std::cout << "\n"
              << "Parameters: scale xm > 0 (minimum value), shape alpha > 0.\n"
              << "  Support: [xm, inf).  Mean = xm*alpha/(alpha-1) for alpha > 1.\n"
              << "  Follows a power law: P(X > x) = (xm/x)^alpha.\n"
              << "Use when: modelling heavy-tailed phenomena where extreme values\n"
              << "          occur with non-negligible frequency (wealth, city sizes,\n"
              << "          network traffic bursts, insurance claims).\n"
              << "\n"
              << "Scenario: File sizes follow Pareto(xm=1 MB, alpha=1.5).\n"
              << "What fraction of files exceed 10 MB?\n";
    auto files = stats::ParetoDistribution::create(1.0, 1.5).unwrap();
    std::cout << std::setprecision(4);
    std::cout << "  P(size > 10 MB)     = " << (1.0 - files.getCumulativeProbability(10.0)) << "\n";
    std::cout << "  Mean file size      = " << files.getMean() << " MB\n";
    std::cout << "  90th percentile     = " << files.getQuantile(0.90) << " MB\n";

    // --- Rayleigh -----------------------------------------------------------
    subsection("Rayleigh distribution");
    std::cout << "\n"
              << "Parameter: sigma > 0.  Support [0, inf).\n"
              << "  Arises as the magnitude of a 2D vector whose components are\n"
              << "  independent Gaussian(0, sigma). Mean = sigma * sqrt(pi/2).\n"
              << "Use when: working with signal amplitude, wind speed (speed of a\n"
              << "          2D wind vector), wave heights, or 2D error radii.\n"
              << "\n"
              << "Scenario: A GPS fix has east and north errors each ~ N(0, 5 m).\n"
              << "The radial error magnitude follows Rayleigh(sigma=5 m).\n"
              << "What is the 95th percentile radial error (2DRMS equivalent)?\n";
    auto gps_error = stats::RayleighDistribution::create(5.0).unwrap();
    std::cout << std::setprecision(2);
    std::cout << "  Mean radial error   = " << gps_error.getMean() << " m\n";
    std::cout << "  95th percentile     = " << gps_error.getQuantile(0.95) << " m\n";

    // --- Erlang -------------------------------------------------------------
    subsection("Erlang distribution");
    std::cout << "\n"
              << "Parameters: integer shape k >= 1, rate lambda > 0.\n"
              << "  Erlang(k, lambda) IS Gamma(k, lambda) with k restricted to\n"
              << "  integers -- the waiting time for exactly k Poisson events.\n"
              << "Use when: the k-stages interpretation is literal (queueing theory,\n"
              << "          telecom traffic, multi-stage service processes) and you\n"
              << "          want the integer constraint enforced by the type.\n"
              << "\n"
              << "Scenario: A packet passes through 4 routers, each adding an\n"
              << "Exponential(2 ms^-1) delay. Total delay ~ Erlang(4, 2).\n";
    auto delay = stats::ErlangDistribution::create(4, 2.0).unwrap();
    std::cout << std::setprecision(3);
    std::cout << "  Mean total delay    = " << delay.getMean() << " ms\n";
    std::cout << "  99th percentile     = " << delay.getQuantile(0.99) << " ms  (SLO candidate)\n";

    // --- Inverse-Gamma ------------------------------------------------------
    subsection("Inverse-Gamma distribution");
    std::cout << "\n"
              << "Parameters: shape alpha > 0, scale beta > 0.\n"
              << "  If X ~ Gamma(alpha, beta) then 1/X ~ InverseGamma(alpha, beta).\n"
              << "  Mean = beta/(alpha-1) for alpha > 1.\n"
              << "Use when: modelling an unknown VARIANCE in Bayesian inference --\n"
              << "          it is the conjugate prior for a Gaussian variance, so the\n"
              << "          posterior stays Inverse-Gamma after observing data.\n"
              << "\n"
              << "Scenario: A Bayesian model's posterior on a noise variance is\n"
              << "InverseGamma(alpha=6, beta=10). Point estimate and 95% interval:\n";
    auto noise_var = stats::InverseGammaDistribution::create(6.0, 10.0).unwrap();
    std::cout << std::setprecision(3);
    std::cout << "  Posterior mean      = " << noise_var.getMean() << "\n";
    std::cout << "  95% interval        = [" << noise_var.getQuantile(0.025) << ", "
              << noise_var.getQuantile(0.975) << "]\n";

    // --- Fisher F -----------------------------------------------------------
    subsection("Fisher F distribution");
    std::cout << "\n"
              << "Parameters: numerator df d1 > 0, denominator df d2 > 0.\n"
              << "  The ratio of two independent scaled chi-squared variables --\n"
              << "  equivalently, the ratio of two sample variances under H0.\n"
              << "Use when: comparing variances, ANOVA F-tests, or testing nested\n"
              << "          regression models.\n"
              << "\n"
              << "Scenario: An ANOVA compares 6 group means with 5 numerator and 30\n"
              << "denominator degrees of freedom. The 5% critical value under H0:\n";
    auto f_stat = stats::FisherF::create(5.0, 30.0).unwrap();
    std::cout << std::setprecision(3);
    std::cout << "  F_{0.95, 5, 30}     = " << f_stat.getQuantile(0.95) << "\n";
    std::cout << "  P(F > 3.0 | H0)     = " << (1.0 - f_stat.getCumulativeProbability(3.0))
              << "  (p-value for an observed F of 3.0)\n";

    // --- Half-Normal --------------------------------------------------------
    subsection("Half-Normal distribution");
    std::cout << "\n"
              << "Parameter: sigma > 0.  Support [0, inf).\n"
              << "  The absolute value of a Gaussian(0, sigma) variable.\n"
              << "  Mean = sigma * sqrt(2/pi).\n"
              << "Use when: only the magnitude of a symmetric error matters\n"
              << "          (absolute deviations, distances from a target), or as a\n"
              << "          weakly-informative prior for scale parameters.\n"
              << "\n"
              << "Scenario: A CNC mill's positioning error along one axis is\n"
              << "N(0, 0.02 mm); the spec cares only about |error|.\n";
    auto abs_err = stats::HalfNormalDistribution::create(0.02).unwrap();
    std::cout << std::setprecision(4);
    std::cout << "  Mean |error|        = " << abs_err.getMean() << " mm\n";
    std::cout << "  P(|error| > 0.05)   = " << (1.0 - abs_err.getCumulativeProbability(0.05))
              << "\n";

    // --- Within-family comparison -------------------------------------------
    subsection("Choosing within the positive-support family");
    std::cout << "\n"
              << "All eleven share (0, inf) support (Pareto: [xm, inf)). Use:\n"
              << "  Waiting for one Poisson event?                 -> Exponential\n"
              << "  Waiting for k events (k integer, literal)?     -> Erlang\n"
              << "  Waiting for k events or flexible skewed shape? -> Gamma\n"
              << "  Sum of squared standard normals?               -> Chi-squared\n"
              << "  Multiplicative process (income, latency)?      -> LogNormal\n"
              << "  Component lifetime with changing hazard rate?  -> Weibull\n"
              << "  Power-law / heavy-tailed quantity?             -> Pareto\n"
              << "  Magnitude of a 2D Gaussian vector?             -> Rayleigh\n"
              << "  Magnitude of a 1D Gaussian variable?           -> Half-Normal\n"
              << "  Unknown variance (Bayesian conjugate prior)?   -> Inverse-Gamma\n"
              << "  Ratio of variances (ANOVA, F-tests)?           -> Fisher F\n"
              << "\n"
              << "Verification: Gamma(1, beta) == Exponential(beta), and\n"
              << "Erlang(k, lambda) == Gamma(k, lambda) for integer k:\n";
    auto gamma_1_2 = stats::GammaDistribution::create(1.0, 2.0).unwrap();
    auto expo_2 = stats::ExponentialDistribution::create(2.0).unwrap();
    auto gamma_4_2 = stats::GammaDistribution::create(4.0, 2.0).unwrap();
    std::cout << std::setprecision(6);
    std::cout << "  Gamma(1,2)  PDF(0.5) = " << gamma_1_2.getProbability(0.5) << "\n";
    std::cout << "  Expo(2)     PDF(0.5) = " << expo_2.getProbability(0.5) << "  (identical)\n";
    std::cout << "  Gamma(4,2)  PDF(2.0) = " << gamma_4_2.getProbability(2.0) << "\n";
    std::cout << "  Erlang(4,2) PDF(2.0) = " << delay.getProbability(2.0) << "  (identical)\n";
}

// ==========================================================================
// FAMILY 3: Bounded continuous
// ==========================================================================
//
// Use these distributions when the quantity is constrained to a finite
// interval -- a proportion, a probability, a bounded physical measurement.
//
//   Uniform          -- maximum entropy on [a, b]; use when every value in
//                       the range is equally plausible and you have no
//                       other information.
//   Beta             -- flexible shape on [0, 1]; use when you have
//                       information about a proportion or probability
//                       (Bayesian prior, click-through rates, success
//                       fractions, mixture weights).
//   TruncatedNormal  -- a Gaussian clipped to [a, b]; use when the
//                       underlying process is Gaussian but the observable
//                       range is physically or procedurally bounded.
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
    auto task = stats::UniformDistribution::create(3.0, 7.0).unwrap();
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
    auto posterior = stats::BetaDistribution::create(7.0, 15.0).unwrap();
    std::cout << "  Posterior mean         = " << std::setprecision(4) << posterior.getMean()
              << "  (point estimate)\n";
    std::cout << "  95% credible interval  = [" << posterior.getQuantile(0.025) << ", "
              << posterior.getQuantile(0.975) << "]\n";
    std::cout << "  P(true rate > 0.5)     = " << (1.0 - posterior.getCumulativeProbability(0.5))
              << "\n";

    // --- Truncated Normal ---------------------------------------------------
    subsection("Truncated Normal distribution");
    std::cout << "\n"
              << "Parameters: mu, sigma of the parent Gaussian, plus bounds [a, b].\n"
              << "  The parent Gaussian's density, renormalized over [a, b].\n"
              << "  Note: mu is the PARENT's mean -- truncation shifts the actual\n"
              << "  mean toward the interval's interior.\n"
              << "Use when: a Gaussian process is observed through a hard limit --\n"
              << "          sensor saturation, inspection cutoffs, physical walls.\n"
              << "\n"
              << "Scenario: Resistors are manufactured with resistance N(100, 3) ohms,\n"
              << "but only parts inside the +/-5% tolerance band [95, 105] are shipped.\n"
              << "What does the shipped population look like?\n";
    auto shipped = stats::TruncatedNormalDistribution::create(100.0, 3.0, 95.0, 105.0).unwrap();
    std::cout << std::setprecision(4);
    std::cout << "  Shipped mean           = " << shipped.getMean() << " ohms\n";
    std::cout << "  Shipped std dev        = " << stats::getStandardDeviation(shipped)
              << "  (< 3: truncation removes the tails)\n";
    std::cout << "  P(shipped < 97 ohms)   = " << shipped.getCumulativeProbability(97.0) << "\n";

    // --- Within-family comparison -------------------------------------------
    subsection("Uniform vs. Beta: the role of prior information");
    std::cout << "\n"
              << "Both have [0,1] support but very different shapes.\n"
              << "Beta(1,1) IS Uniform -- verifying this is a useful sanity check:\n";
    auto beta_1_1 = stats::BetaDistribution::create(1.0, 1.0).unwrap();
    auto uniform_01 = stats::UniformDistribution::create(0.0, 1.0).unwrap();
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
// Use these distributions when the outcome is a count or integer.
//
//   Poisson          -- events at constant rate; mean == variance.
//   Discrete         -- uniform over a finite integer range.
//   Bernoulli        -- a single yes/no trial; the atom of the family.
//   Binomial         -- count of successes in n independent trials.
//   Geometric        -- failures before the first success; memoryless.
//   NegativeBinomial -- failures before r successes; overdispersed counts.
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
    auto calls = stats::PoissonDistribution::create(6.0).unwrap();
    double p_over = 1.0 - calls.getCumulativeProbability(9);
    std::cout << std::setprecision(4);
    std::cout << "  lambda = 6 calls / 30 min\n";
    std::cout << "  P(calls >= 10)   = " << p_over << "\n";
    std::cout << "  P(calls = 6)     = " << calls.getProbability(6) << "  (PMF at mode)\n";
    std::cout << "  90th percentile  = " << static_cast<int>(calls.getQuantile(0.90)) << " calls\n";

    // --- Discrete uniform ---------------------------------------------------
    subsection("Discrete uniform distribution");
    std::cout << "\n"
              << "Parameters: integer bounds [min, max].  Mean = (min+max)/2.\n"
              << "Use when: every integer in the range is equally likely -- a fair\n"
              << "          die roll, a random selection from a numbered list.\n"
              << "\n"
              << "Scenario: Fair six-sided die -- expected value and P(roll >= 5):\n";
    auto die = stats::DiscreteDistribution::create(1, 6).unwrap();
    std::cout << "  Expected value    = " << die.getMean() << "\n";
    std::cout << "  P(roll >= 5)      = " << (1.0 - die.getCumulativeProbability(4)) << "\n";
    std::cout << "  P(roll = 3)       = " << die.getProbability(3) << "  (1/6 for all faces)\n";

    // --- Bernoulli ----------------------------------------------------------
    subsection("Bernoulli distribution");
    std::cout << "\n"
              << "Parameter: success probability p in [0, 1].  Support {0, 1}.\n"
              << "  The single-trial building block: Binomial(1, p) exactly.\n"
              << "  Mean = p; Variance = p*(1-p).\n"
              << "Use when: modelling one yes/no outcome directly -- one conversion,\n"
              << "          one defective part, one coin flip. For n repeated trials\n"
              << "          use Binomial; Bernoulli keeps single-event code honest.\n"
              << "\n"
              << "Scenario: A single visitor converts with probability p = 0.3.\n";
    auto visit = stats::BernoulliDistribution::create(0.3).unwrap();
    std::cout << std::setprecision(4);
    std::cout << "  P(converts)       = " << visit.getProbability(1) << "\n";
    std::cout << "  P(does not)       = " << visit.getProbability(0) << "\n";
    std::cout << "  Variance          = " << visit.getVariance() << "  (maximal at p=0.5)\n";

    // --- Binomial -----------------------------------------------------------
    subsection("Binomial distribution");
    std::cout << "\n"
              << "Parameters: n (number of trials, integer >= 1), p (success prob in [0,1]).\n"
              << "  Models the number of successes in n independent Bernoulli(p) trials.\n"
              << "  Mean = n*p; Variance = n*p*(1-p).\n"
              << "Use when: you know n in advance and want to count successes\n"
              << "          (passed tests, defective items, ad clicks).\n"
              << "\n"
              << "Scenario: A quality inspector tests 20 items from a batch with a\n"
              << "2% defect rate. What is the probability of finding >= 2 defects?\n";
    auto inspect = stats::BinomialDistribution::create(20, 0.02).unwrap();
    std::cout << std::setprecision(4);
    std::cout << "  Mean defects        = " << inspect.getMean() << "\n";
    std::cout << "  P(defects >= 2)     = " << (1.0 - inspect.getCumulativeProbability(1.0))
              << "\n";
    std::cout << "  P(defects = 0)      = " << inspect.getProbability(0) << "  (all pass)\n";

    // --- Geometric ----------------------------------------------------------
    subsection("Geometric distribution");
    std::cout << "\n"
              << "Parameter: success probability p in (0, 1].  Support {0, 1, 2, ...}.\n"
              << "  Counts FAILURES before the first success.  Mean = (1-p)/p.\n"
              << "  The discrete analogue of Exponential: memoryless -- past failures\n"
              << "  say nothing about how many more are coming.\n"
              << "Use when: modelling retries until success (network requests,\n"
              << "          job applications), or gaps between rare events.\n"
              << "\n"
              << "Scenario: An API call succeeds with p = 0.85 per attempt. How many\n"
              << "retries should a client budget for?\n";
    auto retry = stats::GeometricDistribution::create(0.85).unwrap();
    std::cout << std::setprecision(4);
    std::cout << "  Mean failures       = " << retry.getMean() << "\n";
    std::cout << "  P(0 retries needed) = " << retry.getProbability(0) << "\n";
    std::cout << "  P(>= 3 failures)    = " << (1.0 - retry.getCumulativeProbability(2)) << "\n";

    // --- NegativeBinomial ---------------------------------------------------
    subsection("Negative Binomial distribution");
    std::cout << "\n"
              << "Parameters: r (target successes, real > 0), p (success prob in (0,1)).\n"
              << "  Models the number of failures before r successes.\n"
              << "  Mean = r*(1-p)/p; Variance > Mean (overdispersed relative to Poisson).\n"
              << "Use when: count data shows variance >> mean (overdispersion),\n"
              << "          or when modelling trials until a target number of successes.\n"
              << "\n"
              << "Scenario: A recruiter needs to make 5 successful hires (r=5) from\n"
              << "candidates who accept offers with probability p=0.4.\n"
              << "Expected number of rejected offers before 5 hires:\n";
    auto hiring = stats::NegativeBinomialDistribution::create(5.0, 0.4).unwrap();
    std::cout << std::setprecision(2);
    std::cout << "  Mean rejections     = " << hiring.getMean() << "\n";
    std::cout << "  90th percentile     = " << hiring.getQuantile(0.90) << " rejections\n";
    std::cout << std::setprecision(4);
    std::cout << "  P(>= 10 rejections) = " << (1.0 - hiring.getCumulativeProbability(9.0)) << "\n";

    // --- Within-family comparison -------------------------------------------
    subsection("The discrete family is built from Bernoulli");
    std::cout << "\n"
              << "  Bernoulli(p)        == Binomial(1, p)         (one trial)\n"
              << "  Geometric(p)        == NegativeBinomial(1, p) (first success)\n"
              << "  sum of n Bernoullis == Binomial(n, p)\n"
              << "\n"
              << "Verification:\n";
    auto bern = stats::BernoulliDistribution::create(0.3).unwrap();
    auto bin1 = stats::BinomialDistribution::create(1, 0.3).unwrap();
    auto geo = stats::GeometricDistribution::create(0.4).unwrap();
    auto nb1 = stats::NegativeBinomialDistribution::create(1.0, 0.4).unwrap();
    std::cout << "  Bernoulli(0.3) PMF(1) = " << bern.getProbability(1) << "\n";
    std::cout << "  Binomial(1,.3) PMF(1) = " << bin1.getProbability(1) << "  (identical)\n";
    std::cout << "  Geometric(0.4) PMF(2) = " << geo.getProbability(2) << "\n";
    std::cout << "  NegBin(1, 0.4) PMF(2) = " << nb1.getProbability(2) << "  (identical)\n";
}

// ==========================================================================
// FAMILY 5: Circular
// ==========================================================================
//
// Use this distribution when the quantity wraps around: angles, compass
// bearings, time-of-day, phase. Arithmetic on circular quantities requires
// special handling (the midpoint of 350° and 10° is 0°, not 180°).
//
//   Von Mises -- the circular analogue of the Gaussian. Mean direction mu,
//               concentration kappa (0 = uniform, large = concentrated).
// ==========================================================================

void demo_circular() {
    section("Family 5: Circular distributions");

    std::cout << "\n"
              << "Circular distributions model quantities defined on a circle.\n"
              << "Standard distributions cannot be used for angles because the\n"
              << "endpoints wrap around -- distance from 350 degrees to 10 degrees\n"
              << "is 20 degrees, not 340.\n";

    // --- Von Mises ----------------------------------------------------------
    subsection("Von Mises distribution");
    std::cout << "\n"
              << "Parameters: mu in (-pi, pi] (mean direction), kappa >= 0 (concentration).\n"
              << "  kappa = 0   -> uniform on the circle (maximum entropy)\n"
              << "  kappa large -> tightly concentrated around mu\n"
              << "Use when: analysing wind directions, neuron spike phases, compass\n"
              << "          bearings, protein dihedral angles, or any periodic signal.\n"
              << "\n"
              << "Scenario: A wind rose dataset shows prevailing wind from the east\n"
              << "(mu = 0 radians) with moderate concentration (kappa = 2).\n";
    auto wind = stats::VonMisesDistribution::create(0.0, 2.0).unwrap();
    std::cout << std::setprecision(4);
    std::cout << "  Mean direction      = " << wind.getMu() << " radians  (east)\n";
    std::cout << "  Concentration kappa = " << wind.getKappa() << "\n";
    std::cout << "  PDF at mu (peak)    = " << wind.getProbability(0.0) << "\n";
    std::cout << "  PDF at pi (opposite)= " << wind.getProbability(3.14159) << "\n";
    std::cout << "  Entropy             = " << wind.getEntropy() << " nats\n";

    auto uniform_wind = stats::VonMisesDistribution::create(0.0, 0.0).unwrap();
    std::cout << "\nWith kappa=0 (no prevailing direction) PDF is uniform over the circle:\n";
    std::cout << "  PDF at any angle = " << uniform_wind.getProbability(0.0) << "  (= 1/(2*pi))\n";
}

// ==========================================================================
// FAMILY 6: Real-line continuous (asymmetric / heavy-tailed)
// ==========================================================================
//
// Like Family 1 these cover the whole real line, but they are the members
// you reach for when symmetry or thin tails are the WRONG assumption.
//
//   Laplace -- symmetric but sharply peaked with exponential tails;
//              the distribution behind median/L1 methods and DP noise.
//   Cauchy  -- tails so heavy that no mean or variance exists.
//   Gumbel  -- asymmetric; the limiting law of MAXIMA of many draws.
// ==========================================================================

void demo_real_line() {
    section("Family 6: Real-line continuous (asymmetric / heavy-tailed)");

    std::cout << "\n"
              << "Full-real-line distributions for when Gaussian assumptions fail:\n"
              << "outliers dominate, only extremes matter, or moments do not exist.\n";

    // --- Laplace ------------------------------------------------------------
    subsection("Laplace distribution");
    std::cout << "\n"
              << "Parameters: location mu, scale b > 0.  Mean = mu.\n"
              << "  Density falls off as exp(-|x-mu|/b): a sharp peak with tails\n"
              << "  heavier than Gaussian. MLE of mu is the sample MEDIAN.\n"
              << "Use when: residuals are peaked with occasional large outliers\n"
              << "          (L1/robust regression), or adding differential-privacy\n"
              << "          noise (the Laplace mechanism).\n"
              << "\n"
              << "Scenario: A DP query adds Laplace(0, b=2) noise to a count.\n"
              << "How often does the noise exceed +/-5?\n";
    auto dp_noise = stats::LaplaceDistribution::create(0.0, 2.0).unwrap();
    double p_big = 2.0 * (1.0 - dp_noise.getCumulativeProbability(5.0));
    std::cout << std::setprecision(4);
    std::cout << "  P(|noise| > 5)     = " << p_big << "\n";
    std::cout << "  95th percentile    = " << dp_noise.getQuantile(0.95) << "\n";

    // --- Cauchy -------------------------------------------------------------
    subsection("Cauchy distribution");
    std::cout << "\n"
              << "Parameters: location x0 (median), scale gamma > 0.\n"
              << "  The ratio of two independent standard Gaussians; also Student's\n"
              << "  t with nu=1. NO mean or variance exists -- sample averages\n"
              << "  never converge. Quantiles and the median remain well-defined.\n"
              << "Use when: modelling resonance line shapes (physics), ratios of\n"
              << "          centered quantities, or stress-testing methods that\n"
              << "          assume finite moments.\n"
              << "\n"
              << "Scenario: Cauchy(x0=0, gamma=1). Describe it WITHOUT moments:\n";
    auto cauchy = stats::CauchyDistribution::create(0.0, 1.0).unwrap();
    std::cout << "  Median             = " << cauchy.getMedian() << "\n";
    std::cout << "  IQR                = [" << cauchy.getQuantile(0.25) << ", "
              << cauchy.getQuantile(0.75) << "]  (= +/-gamma)\n";
    std::cout << "  P(|X| > 10)        = " << 2.0 * (1.0 - cauchy.getCumulativeProbability(10.0))
              << "  (vs ~0 for Gaussian)\n";

    // --- Gumbel -------------------------------------------------------------
    subsection("Gumbel distribution");
    std::cout << "\n"
              << "Parameters: location mu, scale beta > 0.  Right-skewed.\n"
              << "  The extreme-value limit: the MAXIMUM of many light-tailed draws\n"
              << "  (Gaussian, Exponential, ...) converges to Gumbel.\n"
              << "Use when: modelling annual-maximum floods, peak loads, record\n"
              << "          temperatures -- any block-maximum quantity.\n"
              << "\n"
              << "Scenario: A river's ANNUAL peak level is Gumbel(mu=4 m, beta=0.6).\n"
              << "A levee is designed for the 100-year flood (99th percentile):\n";
    auto flood = stats::GumbelDistribution::create(4.0, 0.6).unwrap();
    std::cout << std::setprecision(3);
    std::cout << "  100-year level     = " << flood.getQuantile(0.99) << " m\n";
    std::cout << "  P(exceed 6 m/yr)   = " << (1.0 - flood.getCumulativeProbability(6.0)) << "\n";

    // --- Within-family comparison -------------------------------------------
    subsection("Choosing within the real-line family");
    std::cout << "\n"
              << "  Symmetric, sharp peak, believable outliers?  -> Laplace\n"
              << "  Extremes of block maxima (floods, records)?  -> Gumbel\n"
              << "  Moments must not be assumed to exist?        -> Cauchy\n"
              << "  Symmetric with mild tails? -> back to Family 1 (Gaussian, t,\n"
              << "  Logistic).\n";
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

    auto gamma = stats::GammaDistribution::create(2.0, 1.0).unwrap();
    gamma.getProbability(xs, out);

    std::cout << "\nGamma(alpha=2, beta=1) PDF across a grid:\n";
    std::cout << std::fixed << std::setprecision(4);
    for (size_t i = 0; i < xs.size(); ++i) {
        std::cout << "  x=" << xs[i] << "  ->  " << out[i] << "\n";
    }
    std::cout << "\nThe same batch interface works identically for all 27 distributions.\n";
}

int main() {
    stats::initialize_performance_systems();

    demo_symmetric_continuous();
    demo_positive_support();
    demo_bounded_continuous();
    demo_discrete();
    demo_circular();
    demo_real_line();
    demo_batch_api();

    std::cout << "\n" << std::string(72, '=') << "\n";
    std::cout << "27 distributions across 7 families.\n";
    std::cout << "See the individual distribution headers for the full API surface.\n";
    std::cout << std::string(72, '=') << "\n";

    return 0;
}

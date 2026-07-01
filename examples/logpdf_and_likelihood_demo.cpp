/**
 * @file logpdf_and_likelihood_demo.cpp
 * @brief Log-probability and log-likelihood workflows in libstats
 *
 * This example demonstrates getLogProbability() — the third primary batch
 * operation alongside getProbability() and getCumulativeProbability() —
 * and shows the real-world patterns that motivate its use:
 *
 *   1. Scalar and batch LogPDF calls
 *   2. Log-likelihood computation from observed data
 *   3. Numerical stability via the log domain (avoiding underflow)
 *   4. Model comparison using log-likelihood
 *   5. Fit-and-score: fit a distribution to observations, then score
 *      new data points using log-probability
 *
 * Why log-probability?
 *   For n independent observations x₁…xₙ the joint likelihood is
 *   ∏ f(xᵢ), which underflows to 0 in floating-point for n ≳ 150
 *   even for well-fitting models. Working in log-space converts the
 *   product to a sum: ∑ log f(xᵢ). This is the standard approach
 *   for MLE, EM algorithms, Bayesian inference, and anomaly scoring.
 */

#define LIBSTATS_FULL_INTERFACE
#include "libstats/libstats.h"

#include <algorithm>  // for std::max_element
#include <cmath>      // for std::exp, std::log, std::isfinite
#include <iomanip>
#include <iostream>
#include <numeric>   // for std::accumulate
#include <random>
#include <span>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
// Utility: compute log-likelihood of a dataset under a distribution.
// Returns sum of log f(xᵢ) for each xᵢ in data.  Values where log f = -inf
// (out-of-support) are excluded from the sum and counted separately.
// ─────────────────────────────────────────────────────────────────────────────
template <typename Dist>
double log_likelihood(const Dist& dist, const std::vector<double>& data) {
    std::vector<double> log_probs(data.size());
    dist.getLogProbability(std::span<const double>(data), std::span<double>(log_probs));

    double ll = 0.0;
    int skipped = 0;
    for (double lp : log_probs) {
        if (std::isfinite(lp)) {
            ll += lp;
        } else {
            ++skipped;
        }
    }
    if (skipped > 0) {
        std::cout << "  [note: " << skipped
                  << " out-of-support points excluded from log-likelihood]\n";
    }
    return ll;
}

// ─────────────────────────────────────────────────────────────────────────────
// Utility: log-sum-exp(v) = log(∑ exp(vᵢ)), computed stably.
// Useful for mixture model likelihoods where the components are in log-space.
// ─────────────────────────────────────────────────────────────────────────────
double log_sum_exp(const std::vector<double>& log_vals) {
    const double max_val = *std::max_element(log_vals.begin(), log_vals.end());
    if (!std::isfinite(max_val)) return max_val;

    double sum = 0.0;
    for (double v : log_vals) {
        sum += std::exp(v - max_val);
    }
    return max_val + std::log(sum);
}

static void section(const char* title) {
    std::cout << "\n" << std::string(60, '=') << "\n"
              << title << "\n"
              << std::string(60, '=') << "\n";
}

int main() {
    stats::initialize_performance_systems();

    std::cout << "=== Log-Probability and Log-Likelihood Workflows ===\n";
    std::cout << "Demonstrating getLogProbability() — the log-space partner\n"
              << "of getProbability() — and its real-world applications.\n";

    std::mt19937 rng(42);

    // ─────────────────────────────────────────────────────────────────────────
    // 1. Scalar log-probability vs log(PDF)
    // ─────────────────────────────────────────────────────────────────────────
    section("1. Scalar Log-Probability");

    auto normal  = stats::GaussianDistribution::create(0.0, 1.0).unwrap();
    auto exp_dist = stats::ExponentialDistribution::create(2.0).unwrap();

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Gaussian N(0,1) at x = 0.0:\n";
    std::cout << "  getProbability(0.0)   = " << normal.getProbability(0.0)
              << "  (PDF value)\n";
    std::cout << "  getLogProbability(0.0) = " << normal.getLogProbability(0.0)
              << "  (log PDF, i.e. log(" << normal.getProbability(0.0) << "))\n\n";

    std::cout << "Exponential(λ=2) at x = 0.5:\n";
    std::cout << "  getProbability(0.5)    = " << exp_dist.getProbability(0.5) << "\n";
    std::cout << "  getLogProbability(0.5) = " << exp_dist.getLogProbability(0.5) << "\n\n";

    // At x=0, Exponential logPDF = log(λ) = log(2) ≈ 0.693
    std::cout << "Exponential(λ=2) at x = 0.0:\n";
    std::cout << "  getLogProbability(0.0) = " << exp_dist.getLogProbability(0.0)
              << "  (= log(λ) = " << std::log(2.0) << ")\n";

    // ─────────────────────────────────────────────────────────────────────────
    // 2. Batch log-probability via span API
    // ─────────────────────────────────────────────────────────────────────────
    section("2. Batch Log-Probability");

    const std::vector<double> xs = {-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0};
    std::vector<double> log_probs(xs.size());

    normal.getLogProbability(std::span<const double>(xs), std::span<double>(log_probs));

    std::cout << "Batch getLogProbability() for Gaussian N(0,1):\n";
    std::cout << "  x     :";
    for (double x : xs) std::cout << std::setw(9) << x;
    std::cout << "\n  LogPDF:";
    for (double lp : log_probs) std::cout << std::setw(9) << std::setprecision(3) << lp;
    std::cout << "\n\n";

    std::cout << "Note: the batch span API and the scalar loop produce identical results;\n"
              << "      the span path routes through the SIMD pipeline for large batches.\n";

    // ─────────────────────────────────────────────────────────────────────────
    // 3. Log-likelihood computation
    // ─────────────────────────────────────────────────────────────────────────
    section("3. Log-Likelihood from Observed Data");

    // Generate 500 observations from the true Gaussian N(1.5, 0.8)
    const double true_mu = 1.5, true_sigma = 0.8;
    auto true_dist = stats::GaussianDistribution::create(true_mu, true_sigma).unwrap();
    auto observations = true_dist.sample(rng, 500);
    std::cout << "Generated 500 observations from Gaussian N(μ=1.5, σ=0.8).\n\n";

    // Evaluate log-likelihood under several candidate models
    auto m1 = stats::GaussianDistribution::create(0.0, 1.0).unwrap();  // wrong params
    auto m2 = stats::GaussianDistribution::create(1.5, 0.8).unwrap();  // correct params
    auto m3 = stats::GaussianDistribution::create(1.5, 2.0).unwrap();  // wrong sigma

    const double ll_m1 = log_likelihood(m1, observations);
    const double ll_m2 = log_likelihood(m2, observations);
    const double ll_m3 = log_likelihood(m3, observations);

    std::cout << std::setprecision(1);
    std::cout << "Log-likelihoods under candidate Gaussian models:\n";
    std::cout << "  N(0.0, 1.0): LL = " << ll_m1 << "  (wrong location)\n";
    std::cout << "  N(1.5, 0.8): LL = " << ll_m2 << "  (correct parameters) ← highest\n";
    std::cout << "  N(1.5, 2.0): LL = " << ll_m3 << "  (over-dispersed)\n\n";

    std::cout << "The model with the highest log-likelihood is the best fit.\n"
              << "ΔLL vs correct model: N(0,1) = " << (ll_m1 - ll_m2)
              << ", N(1.5,2) = " << (ll_m3 - ll_m2) << "\n";

    // ─────────────────────────────────────────────────────────────────────────
    // 4. Numerical stability in the log domain
    // ─────────────────────────────────────────────────────────────────────────
    section("4. Numerical Stability: Avoiding Underflow");

    // With 800 observations the raw product P(x₁)·P(x₂)·…·P(x₈₀₀) underflows
    // to 0 even for a well-fitted model.  The log-domain sum never underflows.
    auto obs_large = true_dist.sample(rng, 800);

    double raw_product = 1.0;
    for (double x : obs_large) {
        raw_product *= true_dist.getProbability(x);
    }

    double log_sum = 0.0;
    std::vector<double> lp_large(obs_large.size());
    true_dist.getLogProbability(std::span<const double>(obs_large),
                                std::span<double>(lp_large));
    for (double lp : lp_large) {
        if (std::isfinite(lp)) log_sum += lp;
    }

    std::cout << "800 observations from N(1.5, 0.8):\n";
    std::cout << "  Raw product ∏ f(xᵢ) = " << raw_product
              << "  ← underflows to 0 for n ≳ 150\n";
    std::cout << "  Log-sum ∑ log f(xᵢ) = " << std::setprecision(1) << log_sum
              << "  ← always finite and meaningful\n\n";

    std::cout << "Use getLogProbability() whenever you multiply more than ~100 densities.\n";

    // ─────────────────────────────────────────────────────────────────────────
    // 5. Model comparison: Gaussian vs Exponential
    // ─────────────────────────────────────────────────────────────────────────
    section("5. Model Comparison via Log-Likelihood");

    // Generate positive-valued latency data from an Exponential distribution
    auto latency_src = stats::ExponentialDistribution::create(0.5).unwrap();  // mean = 2s
    auto latency_obs = latency_src.sample(rng, 300);
    std::cout << "Generated 300 latency observations from Exponential(λ=0.5, mean=2s).\n\n";

    // Fit both models to the data
    auto gauss_fitted = stats::GaussianDistribution::create(1.0, 1.0).unwrap();
    auto exp_fitted   = stats::ExponentialDistribution::create(1.0).unwrap();
    gauss_fitted.fit(latency_obs);
    exp_fitted.fit(latency_obs);

    std::cout << "Fitted parameters:\n";
    std::cout << "  Gaussian:     μ = " << std::setprecision(3) << gauss_fitted.getMean()
              << ",  σ = " << gauss_fitted.getStandardDeviation() << "\n";
    std::cout << "  Exponential:  λ = " << exp_fitted.getLambda()
              << "  (mean = " << exp_fitted.getMean() << ")\n\n";

    const double ll_gauss = log_likelihood(gauss_fitted, latency_obs);
    const double ll_exp   = log_likelihood(exp_fitted,   latency_obs);

    std::cout << std::setprecision(1);
    std::cout << "Log-likelihoods:\n";
    std::cout << "  Gaussian:     LL = " << ll_gauss << "\n";
    std::cout << "  Exponential:  LL = " << ll_exp << "  ← expected to be higher for this data\n\n";

    const char* better = (ll_exp > ll_gauss) ? "Exponential" : "Gaussian";
    std::cout << "Best-fitting model by log-likelihood: " << better << "\n";

    // ─────────────────────────────────────────────────────────────────────────
    // 6. Fit-and-score: anomaly detection via log-probability
    // ─────────────────────────────────────────────────────────────────────────
    section("6. Fit-and-Score: Anomaly Scoring");

    // Fit a model to training data, score test observations.
    // A low log-probability signals an anomalous (unlikely) observation.
    auto training_src = stats::GaussianDistribution::create(10.0, 2.0).unwrap();
    auto training_data = training_src.sample(rng, 500);

    auto scoring_model = stats::GaussianDistribution::create(0.0, 1.0).unwrap();
    scoring_model.fit(training_data);

    std::cout << "Training: 500 observations from N(μ=10, σ=2).\n";
    std::cout << "Fitted model: N(μ=" << std::setprecision(2) << scoring_model.getMean()
              << ", σ=" << scoring_model.getStandardDeviation() << ")\n\n";

    // Score a set of test observations — some normal, some anomalous
    const std::vector<double> test_points = {9.5, 10.0, 10.5, 15.0, 20.0, -5.0};
    std::vector<double> scores(test_points.size());
    scoring_model.getLogProbability(std::span<const double>(test_points),
                                    std::span<double>(scores));

    std::cout << "Anomaly scores (log-probability; lower = more anomalous):\n";
    std::cout << "  " << std::left << std::setw(10) << "x"
              << std::setw(16) << "log P(x|model)" << "Assessment\n";
    std::cout << "  " << std::string(40, '-') << "\n";

    const double threshold = -10.0;  // domain-specific; tune in practice
    for (size_t i = 0; i < test_points.size(); ++i) {
        const char* tag = (scores[i] < threshold) ? "ANOMALY" : "normal";
        std::cout << "  " << std::setw(10) << std::setprecision(1) << test_points[i]
                  << std::setw(16) << std::setprecision(2) << scores[i]
                  << tag << "\n";
    }

    std::cout << "\nKey points:\n"
              << "  \u2022 getLogProbability() is available for all 19 distributions\n"
              << "  • Batch span API routes through SIMD for large inputs\n"
              << "  • Always use log-space when multiplying > ~100 densities\n"
              << "  • Log-likelihood is the natural objective for MLE and model selection\n";

    return 0;
}

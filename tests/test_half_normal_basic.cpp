// Focused unit test for Half-Normal distribution
#include "include/basic_test_runner.h"
#include "include/tests.h"
#include "libstats/distributions/half_normal.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <span>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using namespace std;
using namespace stats;
using namespace stats::tests::fixtures;

int main() {
    BasicTestFormatter::printTestHeader("HalfNormal");

    try {
        // =====================================================================
        // Test 1: Constructors and Destructor
        // =====================================================================
        BasicTestFormatter::printTestStart(1, "Constructors and Destructor");
        cout << "Default σ=1 is the standard Half-Normal. Support: x >= 0." << endl;

        auto default_h = stats::HalfNormalDistribution::create().unwrap();
        BasicTestFormatter::printProperty("Default sigma", default_h.getSigma());

        auto h2 = stats::HalfNormalDistribution::create(2.0).unwrap();
        BasicTestFormatter::printProperty("HN(2) sigma", h2.getSigma());

        auto copy_h = h2;
        BasicTestFormatter::printProperty("Copy sigma", copy_h.getSigma());

        auto temp = stats::HalfNormalDistribution::create(3.0).unwrap();
        auto move_h = std::move(temp);
        BasicTestFormatter::printProperty("Move sigma", move_h.getSigma());

        auto result = HalfNormalDistribution::create(0.5);
        if (result.isOk()) {
            BasicTestFormatter::printProperty("Factory sigma", (*result).getSigma());
        }

        BasicTestFormatter::printTestSuccess("All constructor tests passed");
        BasicTestFormatter::printNewline();

        // =====================================================================
        // Test 2: Parameter Getters and Setters
        // =====================================================================
        BasicTestFormatter::printTestStart(2, "Parameter Getters and Setters");

        // HalfNormal(1): mean = √(2/π) ≈ 0.7979, variance = 1−2/π ≈ 0.3634
        auto h = stats::HalfNormalDistribution::create(1.0).unwrap();
        const double expected_mean = std::sqrt(2.0 / M_PI);
        const double expected_var = 1.0 - 2.0 / M_PI;

        BasicTestFormatter::printProperty("sigma", h.getSigma());
        BasicTestFormatter::printProperty("Mean (expect √(2/π)≈0.7979)", h.getMean());
        BasicTestFormatter::printProperty("Variance (expect 1-2/π≈0.3634)", h.getVariance());
        BasicTestFormatter::printPropertyInt("Num parameters (expect 1)", h.getNumParameters());
        cout << "Name: " << h.getDistributionName() << endl;
        cout << "Is discrete: " << (h.isDiscrete() ? "YES" : "NO") << endl;

        const bool mean_ok = std::abs(h.getMean() - expected_mean) < 1e-10;
        const bool var_ok = std::abs(h.getVariance() - expected_var) < 1e-10;
        cout << "Mean == √(2/π): " << (mean_ok ? "PASS" : "FAIL") << endl;
        cout << "Variance == 1-2/π: " << (var_ok ? "PASS" : "FAIL") << endl;

        h.setSigma(2.0);
        BasicTestFormatter::printProperty("After setSigma(2): sigma", h.getSigma());
        h.setParameters(1.0);
        BasicTestFormatter::printProperty("After setParameters(1): sigma", h.getSigma());

        auto vr = h.trySetSigma(-1.0);
        cout << "trySetSigma(-1) isError: " << (vr.isError() ? "YES" : "NO") << endl;

        if (!mean_ok || !var_ok)
            throw std::runtime_error("Moment accuracy failed");

        BasicTestFormatter::printTestSuccess("All setter/getter tests passed");
        BasicTestFormatter::printNewline();

        // =====================================================================
        // Test 3: Core Probability Methods (known values)
        // =====================================================================
        BasicTestFormatter::printTestStart(3, "Core Probability Methods");
        cout << "HalfNormal(σ=1) references (mpmath dps=40):" << endl;
        cout << "  PDF(0)  = √(2/π) ≈ 0.7979 (the mode)" << endl;
        cout << "  CDF(1)  = erf(1/√2) ≈ 0.6827" << endl;
        cout << "  Median  = √2·erf⁻¹(½) ≈ 0.6745" << endl;

        auto h1 = HalfNormalDistribution::create(1.0).unwrap();

        // PDF(0) = √(2/π) — density is maximal at the origin (support includes 0)
        const double pdf_at_zero = h1.getProbability(0.0);
        const double sqrt_2_over_pi = std::sqrt(2.0 / M_PI);
        BasicTestFormatter::printProperty("PDF(0;σ=1) expect √(2/π)", pdf_at_zero);
        const bool pdf0_ok = std::abs(pdf_at_zero - sqrt_2_over_pi) < 1e-14;
        cout << "PDF(0) == √(2/π): " << (pdf0_ok ? "PASS" : "FAIL") << endl;

        // CDF(1;σ=1) = erf(1/√2) = 0.68268949213708590 (mpmath dps=40)
        const double cdf_at_one = h1.getCumulativeProbability(1.0);
        BasicTestFormatter::printProperty("CDF(1;σ=1) expect 0.682689...", cdf_at_one);
        const bool cdf_ok = std::abs(cdf_at_one - 0.68268949213708590) < 1e-14;
        cout << "CDF(1) == erf(1/√2): " << (cdf_ok ? "PASS" : "FAIL") << endl;

        // CDF(σ) is σ-invariant at x=σ: always erf(1/√2)
        for (double sigma : {0.5, 1.0, 2.0, 5.0}) {
            auto hd = HalfNormalDistribution::create(sigma).unwrap();
            const bool cdf_sigma_ok =
                std::abs(hd.getCumulativeProbability(sigma) - 0.68268949213708590) < 1e-14;
            cout << "CDF(sigma=" << sigma
                 << " at x=sigma) == erf(1/√2): " << (cdf_sigma_ok ? "PASS" : "FAIL") << endl;
        }

        // Out-of-support
        BasicTestFormatter::printProperty("PDF(-1) expect 0", h1.getProbability(-1.0));
        BasicTestFormatter::printProperty("CDF(-1) expect 0", h1.getCumulativeProbability(-1.0));

        // LogPDF consistency
        const double pdf_v = h1.getProbability(2.0);
        const double lp_v = h1.getLogProbability(2.0);
        const bool lp_ok = std::abs(std::log(pdf_v) - lp_v) < 1e-12;
        cout << "log(PDF) == LogPDF: " << (lp_ok ? "PASS" : "FAIL") << endl;

        // Quantile: Q(0.5) = median = 0.67448975019608174 (mpmath dps=40)
        const double q50 = h1.getQuantile(0.5);
        BasicTestFormatter::printProperty("Quantile(0.5) expect 0.674490...", q50);
        const bool q_ok = std::abs(q50 - 0.67448975019608174) < 1e-10;
        cout << "Quantile(0.5) == median: " << (q_ok ? "PASS" : "FAIL") << endl;

        // Utility methods
        BasicTestFormatter::printProperty("Mode = 0", h1.getMode());
        BasicTestFormatter::printProperty("Median (≈0.6745)", h1.getMedian());
        BasicTestFormatter::printProperty("Entropy (≈0.7258)", h1.getEntropy());
        BasicTestFormatter::printProperty("Skewness (≈0.9953)", h1.getSkewness());

        if (!pdf0_ok || !cdf_ok || !lp_ok || !q_ok)
            throw std::runtime_error("Numerical accuracy failed");

        BasicTestFormatter::printTestSuccess("All probability method tests passed");
        BasicTestFormatter::printNewline();

        // =====================================================================
        // Test 4: Random Sampling
        // =====================================================================
        BasicTestFormatter::printTestStart(4, "Random Sampling");
        cout << "Sampling: |Z| with Z ~ Normal(0, σ²), Box–Muller (exact)." << endl;

        mt19937 rng(42);
        auto sample_dist = HalfNormalDistribution::create(1.0).unwrap();
        double s = sample_dist.sample(rng);
        cout << "Single sample >= 0: " << (s >= 0.0 ? "PASS" : "FAIL") << endl;

        const auto samples = sample_dist.sample(rng, 500);
        bool all_nonneg = true;
        double sample_mean = 0.0;
        for (double sv : samples) {
            if (sv < 0.0) {
                all_nonneg = false;
                break;
            }
            sample_mean += sv;
        }
        sample_mean /= static_cast<double>(samples.size());
        cout << "All samples >= 0: " << (all_nonneg ? "PASS" : "FAIL") << endl;
        BasicTestFormatter::printProperty("Sample mean (expect ≈0.798)", sample_mean);
        if (!all_nonneg || std::abs(sample_mean - std::sqrt(2.0 / M_PI)) > 0.1)
            throw std::runtime_error("Sampling failed");

        BasicTestFormatter::printTestSuccess("Sampling tests passed");
        BasicTestFormatter::printNewline();

        // =====================================================================
        // Test 5: Distribution Management (fit, reset)
        // =====================================================================
        BasicTestFormatter::printTestStart(5, "Distribution Management");
        cout << "MLE: σ̂ = √(Σxᵢ²/n). Single pass, no iteration." << endl;

        auto fit_dist = HalfNormalDistribution::create(1.0).unwrap();
        auto source = HalfNormalDistribution::create(3.0).unwrap();
        const auto fit_data = source.sample(rng, 300);
        fit_dist.fit(fit_data);
        BasicTestFormatter::printProperty("Fitted sigma (from HN(3), expect ~3)",
                                          fit_dist.getSigma());

        fit_dist.reset();
        BasicTestFormatter::printProperty("After reset: sigma (expect 1)", fit_dist.getSigma());
        cout << "toString: " << fit_dist.toString() << endl;

        BasicTestFormatter::printTestSuccess("Distribution management tests passed");
        BasicTestFormatter::printNewline();

        // =====================================================================
        // Test 6: Auto-dispatch Batch Operations
        // =====================================================================
        stats::tests::BasicDistConfig cfg{"HalfNormal", {0.1, 0.5, 1.0, 2.0, 3.0}, 0.0, 8.0, 1e-12,
                                          1e-12};
        cfg.invalid_scenarios = {
            {"sigma=-1", [] { return HalfNormalDistribution::create(-1.0).isError(); }},
            {"sigma=0", [] { return HalfNormalDistribution::create(0.0).isError(); }},
            {"sigma=NaN",
             [] {
                 return HalfNormalDistribution::create(std::numeric_limits<double>::quiet_NaN())
                     .isError();
             }},
            {"sigma=inf",
             [] {
                 return HalfNormalDistribution::create(std::numeric_limits<double>::infinity())
                     .isError();
             }},
        };
        auto batch_dist = HalfNormalDistribution::create(1.0).unwrap();
        stats::tests::runBatchTests(cfg, batch_dist);

        BasicTestFormatter::printTestStart(7, "Comparison and Stream Operators");

        auto d1 = HalfNormalDistribution::create(2.0).unwrap();
        auto d2 = HalfNormalDistribution::create(2.0).unwrap();
        auto d3 = HalfNormalDistribution::create(3.0).unwrap();
        cout << "d1 == d2: " << (d1 == d2 ? "true" : "false") << endl;
        cout << "d1 == d3: " << (d1 == d3 ? "true" : "false") << endl;
        stringstream ss;
        ss << d1;
        cout << "Stream output: " << ss.str() << endl;
        auto in_dist = HalfNormalDistribution::create().unwrap();
        ss.seekg(0);
        if (ss >> in_dist)
            cout << "Stream round-trip sigma: " << in_dist.getSigma() << endl;

        BasicTestFormatter::printTestSuccess("Comparison and stream tests passed");
        BasicTestFormatter::printNewline();

        // =====================================================================
        // Test 8: Error Handling
        // =====================================================================
        stats::tests::runErrorTests(cfg);

        BasicTestFormatter::printTestSuccess("All HalfNormal tests completed successfully");
        return 0;

    } catch (const exception& e) {
        cerr << "Test failed: " << e.what() << endl;
        return 1;
    }
}

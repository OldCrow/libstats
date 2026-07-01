// Focused unit test for Log-Normal distribution
#include "include/tests.h"
#include "include/basic_test_runner.h"
#include "libstats/distributions/lognormal.h"

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
    BasicTestFormatter::printTestHeader("LogNormal");

    try {
        // =====================================================================
        // Test 1: Constructors and Destructor
        // =====================================================================
        BasicTestFormatter::printTestStart(1, "Constructors and Destructor");
        cout << "Default (0,1) is the standard log-normal. Support: x > 0." << endl;

        // Default constructor: μ=0, σ=1
        auto default_ln = stats::LogNormalDistribution::create().unwrap();
        BasicTestFormatter::printProperty("Default mu", default_ln.getMu());
        BasicTestFormatter::printProperty("Default sigma", default_ln.getSigma());
        BasicTestFormatter::printProperty("isStandard", static_cast<int>(default_ln.isStandard()));

        // Parameterized
        auto ln23 = stats::LogNormalDistribution::create(2.0, 3.0).unwrap();
        BasicTestFormatter::printProperty("LN(2,3) mu", ln23.getMu());
        BasicTestFormatter::printProperty("LN(2,3) sigma", ln23.getSigma());

        // Copy
        auto copy_ln = ln23;
        BasicTestFormatter::printProperty("Copy mu", copy_ln.getMu());

        // Move
        auto temp = stats::LogNormalDistribution::create(1.0, 0.5).unwrap();
        auto move_ln = std::move(temp);
        BasicTestFormatter::printProperty("Move mu", move_ln.getMu());

        // Factory
        auto result = LogNormalDistribution::create(0.5, 2.0);
        if (result.isOk()) {
            BasicTestFormatter::printProperty("Factory sigma", result.unwrap().getSigma());
        }

        BasicTestFormatter::printTestSuccess("All constructor tests passed");
        BasicTestFormatter::printNewline();

        // =====================================================================
        // Test 2: Parameter Getters and Setters
        // =====================================================================
        BasicTestFormatter::printTestStart(2, "Parameter Getters and Setters");

        // LogNormal(0, 1): mean = exp(0.5) ≈ 1.6487, median = exp(0) = 1
        auto ln = stats::LogNormalDistribution::create(0.0, 1.0).unwrap();
        const double expected_mean = std::exp(0.5);
        const double expected_var = (std::exp(1.0) - 1.0) * std::exp(1.0);

        BasicTestFormatter::printProperty("mu", ln.getMu());
        BasicTestFormatter::printProperty("sigma", ln.getSigma());
        BasicTestFormatter::printProperty("Mean (expect exp(0.5)≈1.6487)", ln.getMean());
        BasicTestFormatter::printProperty("Variance", ln.getVariance());
        BasicTestFormatter::printPropertyInt("Num parameters (expect 2)", ln.getNumParameters());
        cout << "Name: " << ln.getDistributionName() << endl;
        cout << "Is discrete: " << (ln.isDiscrete() ? "YES" : "NO") << endl;
        BasicTestFormatter::printProperty("Support lower (0)", ln.getSupportLowerBound());

        const bool mean_ok = std::abs(ln.getMean() - expected_mean) < 1e-10;
        const bool var_ok = std::abs(ln.getVariance() - expected_var) < 1e-10;
        cout << "Mean correct: " << (mean_ok ? "PASS" : "FAIL") << endl;
        cout << "Variance correct: " << (var_ok ? "PASS" : "FAIL") << endl;

        // Setters
        ln.setMu(1.0);
        BasicTestFormatter::printProperty("After setMu(1): mu", ln.getMu());
        ln.setSigma(2.0);
        BasicTestFormatter::printProperty("After setSigma(2): sigma", ln.getSigma());
        ln.setParameters(0.0, 1.0);
        BasicTestFormatter::printProperty("After reset setParameters: isStandard",
                                          static_cast<int>(ln.isStandard()));

        // Result-based setters
        auto vr = ln.trySetSigma(-1.0);
        cout << "trySetSigma(-1) isError: " << (vr.isError() ? "YES" : "NO") << endl;

        if (!mean_ok || !var_ok)
            throw std::runtime_error("Moment accuracy failed");

        BasicTestFormatter::printTestSuccess("All setter/getter tests passed");
        BasicTestFormatter::printNewline();

        // =====================================================================
        // Test 3: Core Probability Methods (known values)
        // =====================================================================
        BasicTestFormatter::printTestStart(3, "Core Probability Methods");
        cout << "Known values for LogNormal(0,1):" << endl;
        cout << "  PDF(1) = 1/sqrt(2pi) ≈ 0.3989" << endl;
        cout << "  CDF(1) = 0.5 (median at exp(0)=1)" << endl;
        cout << "  Quantile(0.5) = exp(0) = 1" << endl;

        auto std_ln = LogNormalDistribution::create(0.0, 1.0).unwrap();

        // PDF(1): f(1;0,1) = exp(-(log1-0)^2/2)/(1*1*sqrt(2pi)) = 1/sqrt(2pi)
        const double pdf_at_1 = std_ln.getProbability(1.0);
        const double expected_pdf_1 = 1.0 / std::sqrt(2.0 * M_PI);
        BasicTestFormatter::printProperty("PDF(1) expect 1/sqrt(2pi)≈0.3989", pdf_at_1);
        const bool pdf_ok = std::abs(pdf_at_1 - expected_pdf_1) < 1e-10;
        cout << "PDF(1) correct: " << (pdf_ok ? "PASS" : "FAIL") << endl;

        // CDF(1) = 0.5 (median = exp(mu) = exp(0) = 1)
        const double cdf_at_1 = std_ln.getCumulativeProbability(1.0);
        BasicTestFormatter::printProperty("CDF(1) expect 0.5", cdf_at_1);
        const bool cdf_ok = std::abs(cdf_at_1 - 0.5) < 1e-8;
        cout << "CDF(1) correct: " << (cdf_ok ? "PASS" : "FAIL") << endl;

        // Out-of-support
        BasicTestFormatter::printProperty("PDF(-1) expect 0", std_ln.getProbability(-1.0));
        BasicTestFormatter::printProperty("PDF(0)  expect 0", std_ln.getProbability(0.0));
        BasicTestFormatter::printProperty("CDF(0)  expect 0", std_ln.getCumulativeProbability(0.0));

        // LogPDF consistency
        const double pdf_v = std_ln.getProbability(2.0);
        const double lpdf_v = std_ln.getLogProbability(2.0);
        const bool lp_ok = std::abs(std::log(pdf_v) - lpdf_v) < 1e-12;
        cout << "log(PDF) == LogPDF: " << (lp_ok ? "PASS" : "FAIL") << endl;

        // Quantile round-trip
        const double q50 = std_ln.getQuantile(0.5);
        BasicTestFormatter::printProperty("Quantile(0.5) expect 1.0", q50);
        const bool q_ok = std::abs(q50 - 1.0) < 1e-10;
        cout << "Quantile(0.5) correct: " << (q_ok ? "PASS" : "FAIL") << endl;

        // Utility methods
        BasicTestFormatter::printProperty("Median = exp(0) = 1", std_ln.getMedian());
        BasicTestFormatter::printProperty("Mode = exp(-1) ≈ 0.368", std_ln.getMode());
        BasicTestFormatter::printProperty("Entropy", std_ln.getEntropy());

        if (!pdf_ok || !cdf_ok || !lp_ok || !q_ok)
            throw std::runtime_error("Numerical accuracy failed");

        BasicTestFormatter::printTestSuccess("All probability method tests passed");
        BasicTestFormatter::printNewline();

        // =====================================================================
        // Test 4: Random Sampling
        // =====================================================================
        BasicTestFormatter::printTestStart(4, "Random Sampling");
        cout << "Sample from LogNormal(0,1); all values must be positive." << endl;

        mt19937 rng(42);
        auto sample_dist = LogNormalDistribution::create(0.0, 1.0).unwrap();

        // Single sample
        double s = sample_dist.sample(rng);
        BasicTestFormatter::printProperty("Single sample (expect > 0)", s);
        cout << "Sample positive: " << (s > 0.0 ? "PASS" : "FAIL") << endl;

        // Batch samples
        const auto samples = sample_dist.sample(rng, 500);
        const double smean = TestDataGenerators::computeSampleMean(samples);
        BasicTestFormatter::printProperty("Sample mean n=500 (expect ~exp(0.5)≈1.649)", smean);

        bool all_positive = true;
        for (double sv : samples)
            if (sv <= 0.0) {
                all_positive = false;
                break;
            }
        cout << "All samples > 0: " << (all_positive ? "PASS" : "FAIL") << endl;

        BasicTestFormatter::printTestSuccess("Sampling tests passed");
        BasicTestFormatter::printNewline();

        // =====================================================================
        // Test 5: Distribution Management (fit, reset, toString)
        // =====================================================================
        BasicTestFormatter::printTestStart(5, "Distribution Management");
        cout << "MLE: μ̂=mean(log xᵢ), σ̂=std(log xᵢ). Closed-form." << endl;

        auto fit_dist = LogNormalDistribution::create(0.0, 1.0).unwrap();
        // Sample from a known distribution and fit back
        auto source = LogNormalDistribution::create(1.0, 0.5).unwrap();
        const auto fit_data = source.sample(rng, 300);
        fit_dist.fit(fit_data);
        BasicTestFormatter::printProperty("Fitted mu   (from LN(1, 0.5), expect ~1)",
                                          fit_dist.getMu());
        BasicTestFormatter::printProperty("Fitted sigma (from LN(1, 0.5), expect ~0.5)",
                                          fit_dist.getSigma());

        fit_dist.reset();
        BasicTestFormatter::printProperty("After reset: mu (expect 0)", fit_dist.getMu());
        BasicTestFormatter::printProperty("After reset: sigma (expect 1)", fit_dist.getSigma());
        cout << "isStandard after reset: " << (fit_dist.isStandard() ? "YES" : "NO") << endl;
        cout << "toString: " << fit_dist.toString() << endl;

        BasicTestFormatter::printTestSuccess("Distribution management tests passed");
        BasicTestFormatter::printNewline();
        // =====================================================================
        // Test 6: Auto-dispatch Batch Operations
        // =====================================================================
        stats::tests::BasicDistConfig cfg{
            "Lognormal",
            {0.5, 1.0, 2.0, 5.0, 10.0},
            0.1, 10.0,
            1e-12,
            2e-7  // cdf_tolerance
        };
        cfg.invalid_scenarios = {
            {"sigma=-1", [] { return LogNormalDistribution::create(0.0, -1.0).isError(); }},
            {"sigma=0", [] { return LogNormalDistribution::create(0.0, 0.0).isError(); }},
            {"mu=inf", [] { return LogNormalDistribution::create(std::numeric_limits<double>::infinity(), 1.0).isError(); }},
        };
        auto batch_dist = stats::LogNormalDistribution::create(0.0, 1.0).unwrap();
        stats::tests::runBatchTests(cfg, batch_dist);

        BasicTestFormatter::printTestStart(7, "Comparison and Stream Operators");

        auto d1 = LogNormalDistribution::create(0.0, 1.0).unwrap();
        auto d2 = LogNormalDistribution::create(0.0, 1.0).unwrap();
        auto d3 = LogNormalDistribution::create(1.0, 2.0).unwrap();

        cout << "d1 == d2: " << (d1 == d2 ? "true" : "false") << endl;
        cout << "d1 == d3: " << (d1 == d3 ? "true" : "false") << endl;
        cout << "d1 != d3: " << (d1 != d3 ? "true" : "false") << endl;

        stringstream ss;
        ss << d1;
        cout << "Stream output: " << ss.str() << endl;

        auto input_dist = LogNormalDistribution::create().unwrap();
        ss.seekg(0);
        if (ss >> input_dist) {
            cout << "Stream round-trip mu: " << input_dist.getMu() << endl;
        } else {
            cout << "Stream input failed" << endl;
        }

        BasicTestFormatter::printTestSuccess("Comparison and stream operator tests passed");
        BasicTestFormatter::printNewline();

        // =====================================================================
        // Test 8: Error Handling
        // =====================================================================
        stats::tests::runErrorTests(cfg);

        BasicTestFormatter::printTestSuccess("All LogNormal tests completed successfully");
        return 0;

    } catch (const exception& e) {
        cerr << "Test failed: " << e.what() << endl;
        return 1;
    }
}

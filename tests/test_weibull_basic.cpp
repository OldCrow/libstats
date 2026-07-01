// Focused unit test for Weibull distribution
#include "include/tests.h"
#include "include/basic_test_runner.h"
#include "libstats/distributions/weibull.h"

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
    BasicTestFormatter::printTestHeader("Weibull");

    try {
        // =====================================================================
        // Test 1: Constructors and Destructor
        // =====================================================================
        BasicTestFormatter::printTestStart(1, "Constructors and Destructor");
        cout << "Default (1,1) = Exponential(rate=1). Support: x >= 0." << endl;

        auto default_w = stats::WeibullDistribution::create().unwrap();
        BasicTestFormatter::printProperty("Default shape", default_w.getShape());
        BasicTestFormatter::printProperty("Default scale", default_w.getScale());
        BasicTestFormatter::printProperty("isExponential (expect 1)",
                                          static_cast<int>(default_w.isExponential()));

        auto w22 = stats::WeibullDistribution::create(2.0, 2.0).unwrap();
        BasicTestFormatter::printProperty("W(2,2) shape", w22.getShape());
        BasicTestFormatter::printProperty("W(2,2) scale", w22.getScale());

        auto copy_w = w22;
        BasicTestFormatter::printProperty("Copy shape", copy_w.getShape());

        auto temp = stats::WeibullDistribution::create(3.0, 1.5).unwrap();
        auto move_w = std::move(temp);
        BasicTestFormatter::printProperty("Move shape", move_w.getShape());

        auto result = WeibullDistribution::create(0.5, 2.0);
        if (result.isOk()) {
            BasicTestFormatter::printProperty("Factory scale", (*result).getScale());
        }

        BasicTestFormatter::printTestSuccess("All constructor tests passed");
        BasicTestFormatter::printNewline();

        // =====================================================================
        // Test 2: Parameter Getters and Setters
        // =====================================================================
        BasicTestFormatter::printTestStart(2, "Parameter Getters and Setters");

        // Weibull(1,1) = Exponential(rate=1): mean=1, variance=1
        auto w = stats::WeibullDistribution::create(1.0, 1.0).unwrap();
        BasicTestFormatter::printProperty("shape", w.getShape());
        BasicTestFormatter::printProperty("scale", w.getScale());
        BasicTestFormatter::printProperty("Mean (expect 1.0)", w.getMean());
        BasicTestFormatter::printProperty("Variance (expect 1.0)", w.getVariance());
        BasicTestFormatter::printPropertyInt("Num parameters (expect 2)", w.getNumParameters());
        cout << "Name: " << w.getDistributionName() << endl;
        cout << "Is discrete: " << (w.isDiscrete() ? "YES" : "NO") << endl;

        const bool mean_ok = std::abs(w.getMean() - 1.0) < 1e-10;
        const bool var_ok = std::abs(w.getVariance() - 1.0) < 1e-10;
        cout << "Mean == 1.0: " << (mean_ok ? "PASS" : "FAIL") << endl;
        cout << "Variance == 1.0: " << (var_ok ? "PASS" : "FAIL") << endl;

        // Weibull(2,1): mean = Γ(1.5) = √π/2 ≈ 0.8862
        auto w21 = WeibullDistribution::create(2.0, 1.0).unwrap();
        const double expected_mean_21 = std::exp(std::lgamma(1.5));
        const bool mean21_ok = std::abs(w21.getMean() - expected_mean_21) < 1e-10;
        cout << "W(2,1) mean == Gamma(1.5) ≈ 0.8862: " << (mean21_ok ? "PASS" : "FAIL") << endl;

        w.setShape(2.0);
        BasicTestFormatter::printProperty("After setShape(2): shape", w.getShape());
        w.setScale(3.0);
        BasicTestFormatter::printProperty("After setScale(3): scale", w.getScale());
        w.setParameters(1.0, 1.0);
        BasicTestFormatter::printProperty("After reset: isExponential", static_cast<int>(w.isExponential()));

        auto vr = w.trySetShape(-1.0);
        cout << "trySetShape(-1) isError: " << (vr.isError() ? "YES" : "NO") << endl;

        if (!mean_ok || !var_ok || !mean21_ok)
            throw std::runtime_error("Moment accuracy failed");

        BasicTestFormatter::printTestSuccess("All setter/getter tests passed");
        BasicTestFormatter::printNewline();

        // =====================================================================
        // Test 3: Core Probability Methods (known values)
        // =====================================================================
        BasicTestFormatter::printTestStart(3, "Core Probability Methods");
        cout << "Weibull(k=1,λ=1) = Exponential(rate=1):" << endl;
        cout << "  PDF(1) = exp(-1) ≈ 0.3679" << endl;
        cout << "  CDF(1) = 1 - exp(-1) ≈ 0.6321" << endl;
        cout << "  Quantile(0.5) = ln(2) ≈ 0.6931" << endl;

        auto w11 = WeibullDistribution::create(1.0, 1.0).unwrap();

        const double pdf_11_1 = w11.getProbability(1.0);
        const double exp_neg1 = std::exp(-1.0);
        BasicTestFormatter::printProperty("PDF(1;k=1,λ=1) expect exp(-1)", pdf_11_1);
        const bool pdf_ok = std::abs(pdf_11_1 - exp_neg1) < 1e-12;
        cout << "PDF(1) == exp(-1): " << (pdf_ok ? "PASS" : "FAIL") << endl;

        const double cdf_11_1 = w11.getCumulativeProbability(1.0);
        BasicTestFormatter::printProperty("CDF(1;k=1,λ=1) expect 1-exp(-1)", cdf_11_1);
        const bool cdf_ok = std::abs(cdf_11_1 - (1.0 - exp_neg1)) < 1e-12;
        cout << "CDF(1) == 1-exp(-1): " << (cdf_ok ? "PASS" : "FAIL") << endl;

        // CDF at x=scale for any Weibull(k,λ): 1 - exp(-1) regardless of k
        auto w31 = WeibullDistribution::create(3.0, 1.0).unwrap();
        const double cdf_at_scale = w31.getCumulativeProbability(1.0);
        BasicTestFormatter::printProperty("CDF(scale=1;any k) expect 1-1/e", cdf_at_scale);
        cout << "CDF(scale) == 1-1/e: "
             << (std::abs(cdf_at_scale - (1.0 - exp_neg1)) < 1e-12 ? "PASS" : "FAIL") << endl;

        // Out-of-support
        BasicTestFormatter::printProperty("PDF(-1) expect 0", w11.getProbability(-1.0));
        BasicTestFormatter::printProperty("CDF(0) expect 0", w11.getCumulativeProbability(0.0));

        // LogPDF consistency
        const double pdf_v = w11.getProbability(2.0);
        const double lp_v = w11.getLogProbability(2.0);
        const bool lp_ok = std::abs(std::log(pdf_v) - lp_v) < 1e-12;
        cout << "log(PDF) == LogPDF: " << (lp_ok ? "PASS" : "FAIL") << endl;

        // Quantile: Weibull(1,1) → ln(2)
        const double q50 = w11.getQuantile(0.5);
        BasicTestFormatter::printProperty("Quantile(0.5; k=1,λ=1) expect ln(2)≈0.6931", q50);
        cout << "Quantile(0.5) == ln(2): "
             << (std::abs(q50 - std::log(2.0)) < 1e-10 ? "PASS" : "FAIL") << endl;

        // Utility methods
        BasicTestFormatter::printProperty("Mode (k=1: 0)", w11.getMode());
        BasicTestFormatter::printProperty("Median = ln(2)^(1/k)", w11.getMedian());
        BasicTestFormatter::printProperty("Entropy", w11.getEntropy());
        cout << "isExponential (k=1): " << (w11.isExponential() ? "YES" : "NO") << endl;

        if (!pdf_ok || !cdf_ok || !lp_ok)
            throw std::runtime_error("Numerical accuracy failed");

        BasicTestFormatter::printTestSuccess("All probability method tests passed");
        BasicTestFormatter::printNewline();

        // =====================================================================
        // Test 4: Random Sampling
        // =====================================================================
        BasicTestFormatter::printTestStart(4, "Random Sampling");
        cout << "All samples from Weibull(1,1) must be > 0." << endl;

        mt19937 rng(42);
        auto sample_dist = WeibullDistribution::create(1.0, 1.0).unwrap();
        double s = sample_dist.sample(rng);
        cout << "Single sample > 0: " << (s > 0.0 ? "PASS" : "FAIL") << endl;

        const auto samples = sample_dist.sample(rng, 500);
        bool all_pos = true;
        for (double sv : samples)
            if (sv <= 0.0) {
                all_pos = false;
                break;
            }
        cout << "All samples > 0: " << (all_pos ? "PASS" : "FAIL") << endl;

        BasicTestFormatter::printTestSuccess("Sampling tests passed");
        BasicTestFormatter::printNewline();

        // =====================================================================
        // Test 5: Distribution Management (MLE)
        // =====================================================================
        BasicTestFormatter::printTestStart(5, "Distribution Management");
        cout << "MLE: Newton-Raphson on profile score for k; closed-form for λ." << endl;

        auto fit_dist = WeibullDistribution::create(1.0, 1.0).unwrap();
        auto source = WeibullDistribution::create(2.0, 3.0).unwrap();
        const auto fit_data = source.sample(rng, 300);
        fit_dist.fit(fit_data);
        BasicTestFormatter::printProperty("Fitted shape (from W(2,3), expect ~2)",
                                          fit_dist.getShape());
        BasicTestFormatter::printProperty("Fitted scale (from W(2,3), expect ~3)",
                                          fit_dist.getScale());

        fit_dist.reset();
        BasicTestFormatter::printProperty("After reset: shape (expect 1)", fit_dist.getShape());
        BasicTestFormatter::printProperty("After reset: scale (expect 1)", fit_dist.getScale());
        cout << "toString: " << fit_dist.toString() << endl;

        BasicTestFormatter::printTestSuccess("Distribution management tests passed");
        BasicTestFormatter::printNewline();
        // =====================================================================
        // Test 6: Auto-dispatch Batch Operations
        // =====================================================================
        stats::tests::BasicDistConfig cfg{
            "Weibull",
            {0.5, 1.0, 1.5, 2.0, 3.0},
            0.1, 10.0,
            1e-12,
            1e-12
        };
        cfg.invalid_scenarios = {
            {"shape=-1", [] { return WeibullDistribution::create(-1.0, 1.0).isError(); }},
            {"scale=0", [] { return WeibullDistribution::create(1.0, 0.0).isError(); }},
        };
        auto batch_dist = WeibullDistribution::create(2.0, 1.0).unwrap();
        stats::tests::runBatchTests(cfg, batch_dist);

        BasicTestFormatter::printTestStart(7, "Comparison and Stream Operators");

        auto d1 = WeibullDistribution::create(2.0, 1.0).unwrap();
        auto d2 = WeibullDistribution::create(2.0, 1.0).unwrap();
        auto d3 = WeibullDistribution::create(3.0, 2.0).unwrap();
        cout << "d1 == d2: " << (d1 == d2 ? "true" : "false") << endl;
        cout << "d1 == d3: " << (d1 == d3 ? "true" : "false") << endl;
        stringstream ss;
        ss << d1;
        cout << "Stream output: " << ss.str() << endl;
        auto in_dist = WeibullDistribution::create().unwrap();
        ss.seekg(0);
        if (ss >> in_dist)
            cout << "Stream round-trip shape: " << in_dist.getShape() << endl;

        BasicTestFormatter::printTestSuccess("Comparison and stream tests passed");
        BasicTestFormatter::printNewline();

        // =====================================================================
        // Test 8: Error Handling
        // =====================================================================
        stats::tests::runErrorTests(cfg);

        BasicTestFormatter::printTestSuccess("All Weibull tests completed successfully");
        return 0;

    } catch (const exception& e) {
        cerr << "Test failed: " << e.what() << endl;
        return 1;
    }
}

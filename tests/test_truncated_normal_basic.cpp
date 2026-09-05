// Focused unit test for Truncated Normal distribution
#include "include/basic_test_runner.h"
#include "include/tests.h"
#include "libstats/distributions/truncated_normal.h"

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

namespace {
constexpr double kInf = std::numeric_limits<double>::infinity();
}

int main() {
    BasicTestFormatter::printTestHeader("TruncatedNormal");

    try {
        // =====================================================================
        // Test 1: Constructors and Destructor
        // =====================================================================
        BasicTestFormatter::printTestStart(1, "Constructors and Destructor");
        cout << "Default (0, 1, -inf, +inf) degenerates to the plain Gaussian (Z = 1)." << endl;

        auto default_t = stats::TruncatedNormalDistribution::create().unwrap();
        BasicTestFormatter::printProperty("Default mu", default_t.getMu());
        BasicTestFormatter::printProperty("Default Z (expect exactly 1)",
                                          default_t.getNormalizationConstant());

        auto t2 = stats::TruncatedNormalDistribution::create(0.0, 1.0, -2.0, 2.0).unwrap();
        BasicTestFormatter::printProperty("TN(0,1,-2,2) Z (expect 0.9545)",
                                          t2.getNormalizationConstant());

        auto copy_t = t2;
        BasicTestFormatter::printProperty("Copy Z", copy_t.getNormalizationConstant());

        auto temp = stats::TruncatedNormalDistribution::create(1.0, 2.0, 0.0, kInf).unwrap();
        auto move_t = std::move(temp);
        BasicTestFormatter::printProperty("Move mu", move_t.getMu());

        BasicTestFormatter::printTestSuccess("All constructor tests passed");
        BasicTestFormatter::printNewline();

        // =====================================================================
        // Test 2: Parameter Getters and Setters
        // =====================================================================
        BasicTestFormatter::printTestStart(2, "Parameter Getters and Setters");

        auto t = stats::TruncatedNormalDistribution::create(0.0, 1.0, -2.0, 2.0).unwrap();
        BasicTestFormatter::printProperty("mu", t.getMu());
        BasicTestFormatter::printProperty("sigma", t.getSigma());
        BasicTestFormatter::printProperty("a", t.getLowerBound());
        BasicTestFormatter::printProperty("b", t.getUpperBound());
        BasicTestFormatter::printPropertyInt("Num parameters (expect 4)", t.getNumParameters());
        cout << "Name: " << t.getDistributionName() << endl;

        // mpmath dps=40: TN(0,1,-2,2) mean=0, var=0.77374130354992325
        const bool mean_ok = std::abs(t.getMean() - 0.0) < 1e-14;
        const bool var_ok = std::abs(t.getVariance() - 0.77374130354992325) < 1e-13;
        cout << "Mean == 0 (symmetric window): " << (mean_ok ? "PASS" : "FAIL") << endl;
        cout << "Variance == 0.7737413...: " << (var_ok ? "PASS" : "FAIL") << endl;

        t.setMu(0.5);
        BasicTestFormatter::printProperty("After setMu(0.5): mu", t.getMu());
        t.setSigma(2.0);
        BasicTestFormatter::printProperty("After setSigma(2): sigma", t.getSigma());
        t.setLowerBound(-3.0);
        t.setUpperBound(3.0);
        BasicTestFormatter::printProperty("After bound sets: a", t.getLowerBound());
        t.setParameters(0.0, 1.0, -2.0, 2.0);
        BasicTestFormatter::printProperty("After setParameters: Z",
                                          t.getNormalizationConstant());

        auto vr = t.trySetSigma(-1.0);
        cout << "trySetSigma(-1) isError: " << (vr.isError() ? "YES" : "NO") << endl;
        auto vr2 = t.trySetLowerBound(5.0);  // would invert the window (a >= b)
        cout << "trySetLowerBound(5 > b=2) isError: " << (vr2.isError() ? "YES" : "NO") << endl;

        if (!mean_ok || !var_ok)
            throw std::runtime_error("Moment accuracy failed");

        BasicTestFormatter::printTestSuccess("All setter/getter tests passed");
        BasicTestFormatter::printNewline();

        // =====================================================================
        // Test 3: Core Probability Methods (known values, mpmath dps=40)
        // =====================================================================
        BasicTestFormatter::printTestStart(3, "Core Probability Methods");
        cout << "TN(0,1,-2,2) references (mpmath dps=40):" << endl;

        auto t1 = TruncatedNormalDistribution::create(0.0, 1.0, -2.0, 2.0).unwrap();

        const double pdf0 = t1.getProbability(0.0);
        BasicTestFormatter::printProperty("PDF(0) expect 0.4179595502...", pdf0);
        const bool pdf_ok = std::abs(pdf0 - 0.41795955023513457) < 1e-14;
        cout << "PDF(0) matches: " << (pdf_ok ? "PASS" : "FAIL") << endl;

        const double cdf_half = t1.getCumulativeProbability(0.5);
        BasicTestFormatter::printProperty("CDF(0.5) expect 0.7005893286...", cdf_half);
        const bool cdf_ok = std::abs(cdf_half - 0.70058932866297169) < 1e-14;
        cout << "CDF(0.5) matches: " << (cdf_ok ? "PASS" : "FAIL") << endl;

        // Exact bound behaviour
        const bool bounds_ok = t1.getCumulativeProbability(-2.0) == 0.0 &&
                               t1.getCumulativeProbability(2.0) == 1.0 &&
                               t1.getProbability(-2.5) == 0.0 && t1.getProbability(2.5) == 0.0;
        cout << "CDF(a)==0, CDF(b)==1 exactly; PDF outside == 0: "
             << (bounds_ok ? "PASS" : "FAIL") << endl;

        // LogPDF consistency
        const double lp = t1.getLogProbability(1.5);
        const bool lp_ok = std::abs(lp - (-1.9973706209122826)) < 1e-13;
        cout << "LogPDF(1.5) == -1.99737062... : " << (lp_ok ? "PASS" : "FAIL") << endl;

        // Quantile round trip
        const double q05 = t1.getQuantile(0.05);
        BasicTestFormatter::printProperty("Quantile(0.05) expect -1.4722616410...", q05);
        const bool q_ok = std::abs(q05 - (-1.4722616410327654)) < 1e-12;
        cout << "Quantile(0.05) matches: " << (q_ok ? "PASS" : "FAIL") << endl;

        BasicTestFormatter::printProperty("Median (expect 0)", t1.getMedian());
        BasicTestFormatter::printProperty("Mode (expect 0)", t1.getMode());
        BasicTestFormatter::printProperty("Entropy (expect 1.2592412727)", t1.getEntropy());

        if (!pdf_ok || !cdf_ok || !bounds_ok || !lp_ok || !q_ok)
            throw std::runtime_error("Numerical accuracy failed");

        BasicTestFormatter::printTestSuccess("All probability method tests passed");
        BasicTestFormatter::printNewline();

        // =====================================================================
        // Test 4: Random Sampling (inverse-CDF; exact in every regime)
        // =====================================================================
        BasicTestFormatter::printTestStart(4, "Random Sampling");

        mt19937 rng(42);
        auto sample_dist = TruncatedNormalDistribution::create(0.0, 1.0, -2.0, 2.0).unwrap();
        const auto samples = sample_dist.sample(rng, 500);
        bool all_in = true;
        for (double sv : samples)
            if (sv < -2.0 || sv > 2.0) {
                all_in = false;
                break;
            }
        cout << "All samples within [-2, 2]: " << (all_in ? "PASS" : "FAIL") << endl;

        // Far-tail window: naive rejection would essentially never accept
        // (acceptance = Z ~ 2.9e-7); inverse-CDF cost is regime-independent.
        auto tail_dist = TruncatedNormalDistribution::create(0.0, 1.0, 5.0, 6.0).unwrap();
        const auto tail_samples = tail_dist.sample(rng, 500);
        bool tail_in = true;
        for (double sv : tail_samples)
            if (sv < 5.0 || sv > 6.0) {
                tail_in = false;
                break;
            }
        cout << "Far-tail window samples within [5, 6]: " << (tail_in ? "PASS" : "FAIL") << endl;
        if (!all_in || !tail_in)
            throw std::runtime_error("Sampling range failed");

        BasicTestFormatter::printTestSuccess("Sampling tests passed");
        BasicTestFormatter::printNewline();

        // =====================================================================
        // Test 5: Distribution Management (fit, reset)
        // =====================================================================
        BasicTestFormatter::printTestStart(5, "Distribution Management");
        cout << "MLE: bounds KNOWN (held fixed); fixed-point on (mu, sigma)." << endl;

        auto source = TruncatedNormalDistribution::create(0.5, 2.0, -1.0, 4.0).unwrap();
        const auto fit_data = source.sample(rng, 1000);
        auto fit_dist = TruncatedNormalDistribution::create(0.0, 1.0, -1.0, 4.0).unwrap();
        fit_dist.fit(fit_data);
        BasicTestFormatter::printProperty("Fitted mu (true 0.5)", fit_dist.getMu());
        BasicTestFormatter::printProperty("Fitted sigma (true 2.0)", fit_dist.getSigma());

        fit_dist.reset();
        BasicTestFormatter::printProperty("After reset: Z (expect 1)",
                                          fit_dist.getNormalizationConstant());
        cout << "toString: " << fit_dist.toString() << endl;

        BasicTestFormatter::printTestSuccess("Distribution management tests passed");
        BasicTestFormatter::printNewline();

        // =====================================================================
        // Test 6: Auto-dispatch Batch Operations
        // =====================================================================
        stats::tests::BasicDistConfig cfg{
            "TruncatedNormal", {-1.5, -0.5, 0.0, 0.5, 1.5}, -2.0, 2.0, 1e-12, 1e-12};
        cfg.invalid_scenarios = {
            {"sigma=-1",
             [] { return TruncatedNormalDistribution::create(0.0, -1.0, -2.0, 2.0).isError(); }},
            {"a >= b",
             [] { return TruncatedNormalDistribution::create(0.0, 1.0, 2.0, 2.0).isError(); }},
            {"NaN bound",
             [] {
                 return TruncatedNormalDistribution::create(
                            0.0, 1.0, std::numeric_limits<double>::quiet_NaN(), 2.0)
                     .isError();
             }},
            {"window beyond the supported tail (Z underflows)",
             [] { return TruncatedNormalDistribution::create(0.0, 1.0, 40.0, 41.0).isError(); }},
        };
        auto batch_dist = TruncatedNormalDistribution::create(0.0, 1.0, -2.0, 2.0).unwrap();
        stats::tests::runBatchTests(cfg, batch_dist);

        BasicTestFormatter::printTestStart(7, "Comparison and Stream Operators");

        auto d1 = TruncatedNormalDistribution::create(0.0, 1.0, -2.0, 2.0).unwrap();
        auto d2 = TruncatedNormalDistribution::create(0.0, 1.0, -2.0, 2.0).unwrap();
        auto d3 = TruncatedNormalDistribution::create(0.0, 1.0, 0.0, kInf).unwrap();
        cout << "d1 == d2: " << (d1 == d2 ? "true" : "false") << endl;
        cout << "d1 == d3: " << (d1 == d3 ? "true" : "false") << endl;
        auto d4 = TruncatedNormalDistribution::create(0.0, 1.0, 0.0, kInf).unwrap();
        cout << "d3 == d4 (matching +inf bounds): " << (d3 == d4 ? "true" : "false") << endl;
        stringstream ss;
        ss << d1;
        cout << "Stream output: " << ss.str() << endl;
        auto in_dist = TruncatedNormalDistribution::create().unwrap();
        ss.seekg(0);
        if (ss >> in_dist)
            cout << "Stream round-trip Z: " << in_dist.getNormalizationConstant() << endl;

        BasicTestFormatter::printTestSuccess("Comparison and stream tests passed");
        BasicTestFormatter::printNewline();

        // =====================================================================
        // Test 8: Error Handling
        // =====================================================================
        stats::tests::runErrorTests(cfg);

        BasicTestFormatter::printTestSuccess("All TruncatedNormal tests completed successfully");
        return 0;

    } catch (const exception& e) {
        cerr << "Test failed: " << e.what() << endl;
        return 1;
    }
}

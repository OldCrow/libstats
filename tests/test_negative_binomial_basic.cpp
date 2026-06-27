// Focused unit test for Negative Binomial distribution
#include "include/tests.h"
#include "include/basic_test_runner.h"
#include "libstats/distributions/negative_binomial.h"

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
    BasicTestFormatter::printTestHeader("NegativeBinomial");

    try {
        // =====================================================================
        // Test 1: Constructors and Destructor
        // =====================================================================
        BasicTestFormatter::printTestStart(1, "Constructors and Destructor");
        cout << "Default (r=1, p=0.5). Discrete support {0, 1, 2, ...}." << endl;

        auto default_nb = NegativeBinomialDistribution::create().value;
        BasicTestFormatter::printProperty("Default r (expect 1)", default_nb.getR());
        BasicTestFormatter::printProperty("Default p (expect 0.5)", default_nb.getP());

        auto nb1 = NegativeBinomialDistribution::create(3.0, 0.4).value;
        BasicTestFormatter::printProperty("r=3.0", nb1.getR());
        BasicTestFormatter::printProperty("p=0.4", nb1.getP());

        // Real-valued r (key feature over std::negative_binomial_distribution)
        auto nb_real = NegativeBinomialDistribution::create(1.5, 0.6).value;
        BasicTestFormatter::printProperty("r=1.5 (real)", nb_real.getR());

        auto copy_nb = nb1;
        BasicTestFormatter::printProperty("Copy r", copy_nb.getR());

        auto temp = NegativeBinomialDistribution::create(2.0, 0.7).value;
        auto move_nb = std::move(temp);
        BasicTestFormatter::printProperty("Move r (expect 2)", move_nb.getR());

        // Validation
        auto bad1 = NegativeBinomialDistribution::create(0.0, 0.5);
        cout << "r=0 rejected: " << (bad1.isError() ? "YES" : "NO") << endl;
        auto bad2 = NegativeBinomialDistribution::create(-1.0, 0.5);
        cout << "r=-1 rejected: " << (bad2.isError() ? "YES" : "NO") << endl;
        auto bad3 = NegativeBinomialDistribution::create(2.0, 0.0);
        cout << "p=0 rejected: " << (bad3.isError() ? "YES" : "NO") << endl;
        auto bad4 = NegativeBinomialDistribution::create(2.0, 1.1);
        cout << "p=1.1 rejected: " << (bad4.isError() ? "YES" : "NO") << endl;

        if (!bad1.isError() || !bad2.isError() || !bad3.isError() || !bad4.isError())
            throw runtime_error("Bad parameter validation failed");

        BasicTestFormatter::printTestSuccess("All constructor tests passed");
        BasicTestFormatter::printNewline();

        // =====================================================================
        // Test 2: Parameter Getters and Setters
        // =====================================================================
        BasicTestFormatter::printTestStart(2, "Parameter Getters and Setters");

        auto nb = NegativeBinomialDistribution::create(2.0, 0.5).value;
        BasicTestFormatter::printProperty("r", nb.getR());
        BasicTestFormatter::printProperty("p", nb.getP());
        BasicTestFormatter::printPropertyInt("Num parameters (expect 2)", nb.getNumParameters());
        cout << "Name: " << nb.getDistributionName() << endl;
        cout << "isDiscrete: " << (nb.isDiscrete() ? "YES" : "NO") << endl;

        // Moments: NegBinom(2, 0.5) — mean=r(1-p)/p=2, var=r(1-p)/p^2=4
        const double expected_mean = 2.0;
        const double expected_var  = 4.0;
        BasicTestFormatter::printProperty("Mean (expect 2.0)", nb.getMean());
        BasicTestFormatter::printProperty("Variance (expect 4.0)", nb.getVariance());
        BasicTestFormatter::printProperty("Skewness", nb.getSkewness());
        BasicTestFormatter::printProperty("Kurtosis", nb.getKurtosis());

        const bool moments_ok = (std::abs(nb.getMean() - expected_mean) < 1e-12) &&
                                (std::abs(nb.getVariance() - expected_var) < 1e-12);
        cout << "Moments correct (NB(2,0.5)): " << (moments_ok ? "PASS" : "FAIL") << endl;

        nb.setP(0.4);
        BasicTestFormatter::printProperty("After setP(0.4): p", nb.getP());
        nb.setR(3.0);
        BasicTestFormatter::printProperty("After setR(3.0): r", nb.getR());
        nb.setParameters(2.0, 0.5);

        // trySet tests
        auto rv1 = nb.trySetR(-1.0);
        cout << "trySetR(-1) isError: " << (rv1.isError() ? "YES" : "NO") << endl;
        auto rv2 = nb.trySetP(0.0);
        cout << "trySetP(0) isError: " << (rv2.isError() ? "YES" : "NO") << endl;

        // Support: [0, +inf)
        BasicTestFormatter::printProperty("SupportLower (expect 0)", nb.getSupportLowerBound());
        const bool inf_upper = std::isinf(nb.getSupportUpperBound());
        cout << "SupportUpper is +inf: " << (inf_upper ? "YES" : "NO") << endl;

        if (!moments_ok || !rv1.isError() || !rv2.isError())
            throw runtime_error("Parameter test failed");

        BasicTestFormatter::printTestSuccess("All setter/getter tests passed");
        BasicTestFormatter::printNewline();

        // =====================================================================
        // Test 3: Core Probability Methods (known values)
        // =====================================================================
        BasicTestFormatter::printTestStart(3, "Core Probability Methods");
        cout << "NB(2, 0.5): PMF(0)=0.25, PMF(1)=0.25, CDF(0)=0.25, CDF(1)=0.5" << endl;

        auto b = NegativeBinomialDistribution::create(2.0, 0.5).value;

        // PMF(0) = p^r = 0.5^2 = 0.25
        const double pmf0 = b.getProbability(0.0);
        BasicTestFormatter::printProperty("PMF(0) expect 0.25", pmf0);
        const bool pmf0_ok = (std::abs(pmf0 - 0.25) < 1e-10);
        cout << "PMF(0) correct: " << (pmf0_ok ? "PASS" : "FAIL") << endl;

        // PMF(1) = C(2,1)*p^2*(1-p)^1 = 2*0.25*0.5 = 0.25
        const double pmf1 = b.getProbability(1.0);
        BasicTestFormatter::printProperty("PMF(1) expect 0.25", pmf1);
        const bool pmf1_ok = (std::abs(pmf1 - 0.25) < 1e-10);
        cout << "PMF(1) correct: " << (pmf1_ok ? "PASS" : "FAIL") << endl;

        // LogPMF consistency
        const double logpmf0 = b.getLogProbability(0.0);
        const bool logpmf_ok = (std::abs(std::log(pmf0) - logpmf0) < 1e-12);
        BasicTestFormatter::printProperty("LogPMF(0)", logpmf0);
        cout << "log(PMF(0)) == LogPMF(0): " << (logpmf_ok ? "PASS" : "FAIL") << endl;

        // CDF(0) = I_p(r, 1) = I_{0.5}(2, 1) = 0.25
        const double cdf0 = b.getCumulativeProbability(0.0);
        BasicTestFormatter::printProperty("CDF(0) expect 0.25", cdf0);
        const bool cdf0_ok = (std::abs(cdf0 - 0.25) < 1e-8);
        cout << "CDF(0) correct: " << (cdf0_ok ? "PASS" : "FAIL") << endl;

        // CDF(1) = PMF(0)+PMF(1) = 0.5
        const double cdf1 = b.getCumulativeProbability(1.0);
        BasicTestFormatter::printProperty("CDF(1) expect 0.5", cdf1);
        const bool cdf1_ok = (std::abs(cdf1 - 0.5) < 1e-8);
        cout << "CDF(1) correct: " << (cdf1_ok ? "PASS" : "FAIL") << endl;

        // CDF boundary: k < 0 → 0
        const double cdf_neg = b.getCumulativeProbability(-1.0);
        cout << "CDF(-1) = " << cdf_neg << " (expect 0): " << (cdf_neg == 0.0 ? "PASS" : "FAIL") << endl;

        // PMF out of range
        const double pmf_neg = b.getProbability(-1.0);
        cout << "PMF(-1) = 0: " << (pmf_neg == 0.0 ? "PASS" : "FAIL") << endl;

        // Quantile round-trip
        const double q1 = b.getQuantile(cdf1);
        cout << "Quantile(CDF(1)) = " << q1 << " (expect 1): "
             << (std::abs(q1 - 1.0) < 0.5 ? "PASS" : "FAIL") << endl;

        // Mode for NB(2, 0.5): floor((r-1)(1-p)/p) = floor(1*1) = 1
        const double mode = b.getMode();
        BasicTestFormatter::printProperty("Mode (expect 1)", mode);

        if (!pmf0_ok || !pmf1_ok || !logpmf_ok || !cdf0_ok || !cdf1_ok)
            throw runtime_error("Probability accuracy failed");

        BasicTestFormatter::printTestSuccess("All probability method tests passed");
        BasicTestFormatter::printNewline();

        // =====================================================================
        // Test 4: Random Sampling (Gamma-Poisson mixture)
        // =====================================================================
        BasicTestFormatter::printTestStart(4, "Random Sampling");
        cout << "Gamma(r,(1-p)/p)-Poisson mixture. Supports real r." << endl;

        mt19937 rng(42);
        auto sample_nb = NegativeBinomialDistribution::create(2.0, 0.5).value;
        const double s = sample_nb.sample(rng);
        cout << "Single sample ≥ 0: " << (s >= 0.0 ? "PASS" : "FAIL") << endl;

        const auto samples = sample_nb.sample(rng, 500);
        bool all_non_neg = true;
        double sample_mean = 0.0;
        for (double sv : samples) {
            sample_mean += sv;
            if (sv < 0.0) { all_non_neg = false; }
        }
        sample_mean /= 500.0;
        cout << "All 500 samples ≥ 0: " << (all_non_neg ? "PASS" : "FAIL") << endl;
        BasicTestFormatter::printProperty("Sample mean (expect ~2.0)", sample_mean);
        const bool sample_mean_ok = (std::abs(sample_mean - 2.0) < 1.5);
        cout << "Sample mean ≈ 2.0: " << (sample_mean_ok ? "PASS" : "FAIL") << endl;

        // Real-valued r sampling
        auto sample_real = NegativeBinomialDistribution::create(1.5, 0.6).value;
        const double sr = sample_real.sample(rng);
        cout << "Real r=1.5 sample ≥ 0: " << (sr >= 0.0 ? "PASS" : "FAIL") << endl;

        if (!all_non_neg)
            throw runtime_error("Negative sample value");

        BasicTestFormatter::printTestSuccess("Sampling tests passed");
        BasicTestFormatter::printNewline();

        // =====================================================================
        // Test 5: Distribution Management (MLE)
        // =====================================================================
        BasicTestFormatter::printTestStart(5, "Distribution Management");
        cout << "MLE: MoM seed + Newton-Raphson on profile score (digamma/trigamma)." << endl;

        auto fit_dist = NegativeBinomialDistribution::create(1.0, 0.5).value;
        // Fit to NB(3, 0.6) samples
        auto source = NegativeBinomialDistribution::create(3.0, 0.6).value;
        const auto fit_data = source.sample(rng, 500);
        fit_dist.fit(fit_data);
        BasicTestFormatter::printProperty("Fitted r (from NB(3,0.6), expect ~3)", fit_dist.getR());
        BasicTestFormatter::printProperty("Fitted p (from NB(3,0.6), expect ~0.6)", fit_dist.getP());
        const bool fit_ok = (fit_dist.getR() > 0.0) && (fit_dist.getP() > 0.0)
                         && (fit_dist.getP() <= 1.0);
        cout << "Fit parameters valid: " << (fit_ok ? "PASS" : "FAIL") << endl;

        fit_dist.reset();
        BasicTestFormatter::printProperty("After reset: r (expect 1)", fit_dist.getR());
        BasicTestFormatter::printProperty("After reset: p (expect 0.5)", fit_dist.getP());
        cout << "toString: " << fit_dist.toString() << endl;

        BasicTestFormatter::printTestSuccess("Distribution management tests passed");
        BasicTestFormatter::printNewline();
        // =====================================================================
        // Test 6: Auto-dispatch Batch Operations
        // =====================================================================
        stats::tests::BasicDistConfig cfg{
            "Negativebinomial",
            {0.0, 1.0, 2.0, 5.0, 10.0},
            0.0, 19.5,
            1e-12,
            1e-12
        };
        cfg.invalid_scenarios = {
            {"r=0", [] { return NegativeBinomialDistribution::create(0.0, 0.5).isError(); }},
            {"r=-1", [] { return NegativeBinomialDistribution::create(-1.0, 0.5).isError(); }},
            {"p=0", [] { return NegativeBinomialDistribution::create(1.0, 0.0).isError(); }},
            {"p=1.1", [] { return NegativeBinomialDistribution::create(1.0, 1.1).isError(); }},
        };
        auto batch_nb = NegativeBinomialDistribution::create(5.0, 0.4).value;
        stats::tests::runBatchTests(cfg, batch_nb);

        BasicTestFormatter::printTestStart(7, "Comparison and Stream Operators");

        auto d1 = NegativeBinomialDistribution::create(2.0, 0.5).value;
        auto d2 = NegativeBinomialDistribution::create(2.0, 0.5).value;
        auto d3 = NegativeBinomialDistribution::create(3.0, 0.5).value;
        cout << "d1 == d2: " << (d1 == d2 ? "true" : "false") << endl;
        cout << "d1 == d3: " << (d1 == d3 ? "true" : "false") << endl;
        cout << "d1 != d3: " << (d1 != d3 ? "true" : "false") << endl;

        stringstream ss;
        ss << d1;
        cout << "Stream output: " << ss.str() << endl;
        auto in_dist = NegativeBinomialDistribution::create().value;
        ss.seekg(0);
        if (ss >> in_dist)
            cout << "Stream round-trip r=" << in_dist.getR()
                 << " p=" << in_dist.getP() << endl;

        BasicTestFormatter::printTestSuccess("Comparison and stream tests passed");
        BasicTestFormatter::printNewline();

        // =====================================================================
        // Test 8: Error Handling
        // =====================================================================
        stats::tests::runErrorTests(cfg);

        BasicTestFormatter::printTestSuccess("All NegativeBinomial tests completed successfully");
        return 0;

    } catch (const exception& e) {
        cerr << "Test failed: " << e.what() << endl;
        return 1;
    }
}

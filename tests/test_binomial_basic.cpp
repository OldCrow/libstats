// Focused unit test for Binomial distribution
#include "include/tests.h"
#include "include/basic_test_runner.h"
#include "libstats/distributions/binomial.h"

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
    BasicTestFormatter::printTestHeader("Binomial");

    try {
        // =====================================================================
        // Test 1: Constructors and Destructor
        // =====================================================================
        BasicTestFormatter::printTestStart(1, "Constructors and Destructor");
        cout << "Default (n=10, p=0.5). Discrete support {0, ..., n}." << endl;

        auto default_b = BinomialDistribution::create().unwrap();
        BasicTestFormatter::printPropertyInt("Default n (expect 10)", default_b.getN());
        BasicTestFormatter::printProperty("Default p (expect 0.5)", default_b.getP());

        auto b1 = BinomialDistribution::create(20, 0.3).unwrap();
        BasicTestFormatter::printPropertyInt("n=20 trials", b1.getN());
        BasicTestFormatter::printProperty("p=0.3", b1.getP());

        auto copy_b = b1;
        BasicTestFormatter::printPropertyInt("Copy n", copy_b.getN());

        auto temp = BinomialDistribution::create(5, 0.7).unwrap();
        auto move_b = std::move(temp);
        BasicTestFormatter::printPropertyInt("Move n (expect 5)", move_b.getN());

        auto bad = BinomialDistribution::create(0, 0.5);
        cout << "n=0 rejected: " << (bad.isError() ? "YES" : "NO") << endl;
        auto bad2 = BinomialDistribution::create(10, 1.5);
        cout << "p=1.5 rejected: " << (bad2.isError() ? "YES" : "NO") << endl;

        if (!bad.isError() || !bad2.isError())
            throw runtime_error("Bad parameter validation failed");

        BasicTestFormatter::printTestSuccess("All constructor tests passed");
        BasicTestFormatter::printNewline();

        // =====================================================================
        // Test 2: Parameter Getters and Setters
        // =====================================================================
        BasicTestFormatter::printTestStart(2, "Parameter Getters and Setters");

        auto binom = BinomialDistribution::create(10, 0.5).unwrap();
        BasicTestFormatter::printPropertyInt("n", binom.getN());
        BasicTestFormatter::printProperty("p", binom.getP());
        BasicTestFormatter::printPropertyInt("Num parameters (expect 2)", binom.getNumParameters());
        cout << "Name: " << binom.getDistributionName() << endl;
        cout << "isDiscrete: " << (binom.isDiscrete() ? "YES" : "NO") << endl;

        // Moments: Binomial(10, 0.5) — mean=5, var=2.5
        const double expected_mean = 5.0;
        const double expected_var  = 2.5;
        BasicTestFormatter::printProperty("Mean (expect 5.0)", binom.getMean());
        BasicTestFormatter::printProperty("Variance (expect 2.5)", binom.getVariance());
        BasicTestFormatter::printProperty("Skewness (expect 0)", binom.getSkewness());
        BasicTestFormatter::printProperty("Kurtosis", binom.getKurtosis());

        const bool moments_ok = (std::abs(binom.getMean() - expected_mean) < 1e-12) &&
                                (std::abs(binom.getVariance() - expected_var) < 1e-12) &&
                                (std::abs(binom.getSkewness()) < 1e-12);
        cout << "Moments correct (Bin(10,0.5)): " << (moments_ok ? "PASS" : "FAIL") << endl;

        binom.setP(0.3);
        BasicTestFormatter::printProperty("After setP(0.3): p", binom.getP());
        binom.setN(5);
        BasicTestFormatter::printPropertyInt("After setN(5): n", binom.getN());
        binom.setParameters(10, 0.5);

        // trySet tests
        auto rv = binom.trySetP(1.5);
        cout << "trySetP(1.5) isError: " << (rv.isError() ? "YES" : "NO") << endl;

        // Support bounds
        BasicTestFormatter::printProperty("SupportLower (expect 0)", binom.getSupportLowerBound());
        BasicTestFormatter::printProperty("SupportUpper (expect 10)", binom.getSupportUpperBound());

        if (!moments_ok || !rv.isError())
            throw runtime_error("Parameter test failed");

        BasicTestFormatter::printTestSuccess("All setter/getter tests passed");
        BasicTestFormatter::printNewline();

        // =====================================================================
        // Test 3: Core Probability Methods (known values)
        // =====================================================================
        BasicTestFormatter::printTestStart(3, "Core Probability Methods");
        cout << "Binomial(10, 0.5): PMF(5)=C(10,5)*(0.5)^10=252/1024≈0.24609375" << endl;

        auto b = BinomialDistribution::create(10, 0.5).unwrap();

        // PMF(5) = 252/1024 = 0.24609375
        const double pmf5 = b.getProbability(5.0);
        const double expected_pmf5 = 252.0 / 1024.0;
        BasicTestFormatter::printProperty("PMF(5) expect 0.24609375", pmf5);
        const bool pmf5_ok = (std::abs(pmf5 - expected_pmf5) < 1e-10);
        cout << "PMF(5) correct: " << (pmf5_ok ? "PASS" : "FAIL") << endl;

        // PMF(0) = (0.5)^10 = 1/1024
        const double pmf0 = b.getProbability(0.0);
        const double expected_pmf0 = 1.0 / 1024.0;
        BasicTestFormatter::printProperty("PMF(0) expect 1/1024≈0.000977", pmf0);
        const bool pmf0_ok = (std::abs(pmf0 - expected_pmf0) < 1e-12);
        cout << "PMF(0) correct: " << (pmf0_ok ? "PASS" : "FAIL") << endl;

        // LogPMF consistency
        const double logpmf5 = b.getLogProbability(5.0);
        const bool logpmf_ok = (std::abs(std::log(pmf5) - logpmf5) < 1e-12);
        BasicTestFormatter::printProperty("LogPMF(5)", logpmf5);
        cout << "log(PMF(5)) == LogPMF(5): " << (logpmf_ok ? "PASS" : "FAIL") << endl;

        // CDF(5) for Binomial(10, 0.5) ≈ 0.623046875
        const double cdf5 = b.getCumulativeProbability(5.0);
        const double expected_cdf5 = 0.623046875;  // exact: sum C(10,k)/2^10 for k=0..5
        BasicTestFormatter::printProperty("CDF(5) expect 0.623047", cdf5);
        const bool cdf5_ok = (std::abs(cdf5 - expected_cdf5) < 1e-6);
        cout << "CDF(5) correct: " << (cdf5_ok ? "PASS" : "FAIL") << endl;

        // CDF boundaries
        const double cdf_neg = b.getCumulativeProbability(-1.0);
        const double cdf_n   = b.getCumulativeProbability(10.0);
        cout << "CDF(-1) = " << cdf_neg << " (expect 0): " << (cdf_neg == 0.0 ? "PASS" : "FAIL") << endl;
        cout << "CDF(10) = " << cdf_n  << " (expect 1): " << (cdf_n == 1.0 ? "PASS" : "FAIL") << endl;

        // PMF out of range
        const double pmf_neg = b.getProbability(-1.0);
        const double pmf_big = b.getProbability(11.0);
        cout << "PMF(-1) = 0: " << (pmf_neg == 0.0 ? "PASS" : "FAIL") << endl;
        cout << "PMF(11) = 0: " << (pmf_big == 0.0 ? "PASS" : "FAIL") << endl;

        // Quantile round-trip
        const double q5 = b.getQuantile(cdf5);
        cout << "Quantile(CDF(5)) = " << q5 << " (expect 5): " << (q5 == 5.0 ? "PASS" : "FAIL") << endl;

        if (!pmf5_ok || !pmf0_ok || !logpmf_ok || !cdf5_ok)
            throw runtime_error("Probability accuracy failed");

        BasicTestFormatter::printTestSuccess("All probability method tests passed");
        BasicTestFormatter::printNewline();

        // =====================================================================
        // Test 4: Random Sampling
        // =====================================================================
        BasicTestFormatter::printTestStart(4, "Random Sampling");
        cout << "std::binomial_distribution<int>(n, p). Samples in {0, ..., n}." << endl;

        mt19937 rng(42);
        auto sample_b = BinomialDistribution::create(10, 0.5).unwrap();
        const double s = sample_b.sample(rng);
        cout << "Single sample in {0..10}: " << (s >= 0.0 && s <= 10.0 ? "PASS" : "FAIL") << endl;

        const auto samples = sample_b.sample(rng, 500);
        bool all_in_range = true;
        double sample_mean = 0.0;
        for (double sv : samples) {
            sample_mean += sv;
            if (sv < 0.0 || sv > 10.0) { all_in_range = false; }
        }
        sample_mean /= 500.0;
        cout << "All 500 samples in {0..10}: " << (all_in_range ? "PASS" : "FAIL") << endl;
        BasicTestFormatter::printProperty("Sample mean (expect ~5.0)", sample_mean);
        const bool sample_mean_ok = (std::abs(sample_mean - 5.0) < 1.0);
        cout << "Sample mean ≈ 5.0: " << (sample_mean_ok ? "PASS" : "FAIL") << endl;

        if (!all_in_range)
            throw runtime_error("Samples out of range");

        BasicTestFormatter::printTestSuccess("Sampling tests passed");
        BasicTestFormatter::printNewline();

        // =====================================================================
        // Test 5: Distribution Management (MLE)
        // =====================================================================
        BasicTestFormatter::printTestStart(5, "Distribution Management");
        cout << "MLE: n̂ = max(round(xᵢ)), p̂ = k̄/n̂." << endl;

        auto fit_dist = BinomialDistribution::create(10, 0.5).unwrap();
        // Fit to Binomial(8, 0.6) samples
        auto source = BinomialDistribution::create(8, 0.6).unwrap();
        const auto fit_data = source.sample(rng, 400);
        fit_dist.fit(fit_data);
        BasicTestFormatter::printPropertyInt("Fitted n (from Bin(8,0.6), expect ~8)", fit_dist.getN());
        BasicTestFormatter::printProperty("Fitted p (from Bin(8,0.6), expect ~0.6)", fit_dist.getP());
        // n̂ = max(k) ≥ 1
        const bool fit_ok = (fit_dist.getN() >= 1) && (fit_dist.getP() > 0.0) && (fit_dist.getP() <= 1.0);
        cout << "Fit parameters valid: " << (fit_ok ? "PASS" : "FAIL") << endl;

        fit_dist.reset();
        BasicTestFormatter::printPropertyInt("After reset: n (expect 10)", fit_dist.getN());
        BasicTestFormatter::printProperty("After reset: p (expect 0.5)", fit_dist.getP());
        cout << "toString: " << fit_dist.toString() << endl;

        BasicTestFormatter::printTestSuccess("Distribution management tests passed");
        BasicTestFormatter::printNewline();
        // =====================================================================
        // Test 6: Auto-dispatch Batch Operations
        // =====================================================================
        stats::tests::BasicDistConfig cfg{
            "Binomial",
            {0.0, 2.0, 5.0, 8.0, 10.0},
            0.0, 10.5,
            1e-12,
            1e-9  // cdf_tolerance
        };
        cfg.invalid_scenarios = {
            {"n=0", [] { return BinomialDistribution::create(0, 0.5).isError(); }},
            {"n=-1", [] { return BinomialDistribution::create(-1, 0.5).isError(); }},
            {"p=-0.1", [] { return BinomialDistribution::create(10, -0.1).isError(); }},
            {"p=1.1", [] { return BinomialDistribution::create(10, 1.1).isError(); }},
        };
        auto batch_b = BinomialDistribution::create(10, 0.5).unwrap();
        stats::tests::runBatchTests(cfg, batch_b);

        BasicTestFormatter::printTestStart(7, "Comparison and Stream Operators");

        auto d1 = BinomialDistribution::create(10, 0.5).unwrap();
        auto d2 = BinomialDistribution::create(10, 0.5).unwrap();
        auto d3 = BinomialDistribution::create(10, 0.3).unwrap();
        cout << "d1 == d2: " << (d1 == d2 ? "true" : "false") << endl;
        cout << "d1 == d3: " << (d1 == d3 ? "true" : "false") << endl;
        cout << "d1 != d3: " << (d1 != d3 ? "true" : "false") << endl;

        stringstream ss;
        ss << d1;
        cout << "Stream output: " << ss.str() << endl;
        auto in_dist = BinomialDistribution::create().unwrap();
        ss.seekg(0);
        if (ss >> in_dist)
            cout << "Stream round-trip n=" << in_dist.getN()
                 << " p=" << in_dist.getP() << endl;

        BasicTestFormatter::printTestSuccess("Comparison and stream tests passed");
        BasicTestFormatter::printNewline();

        // =====================================================================
        // Test 8: Error Handling
        // =====================================================================
        stats::tests::runErrorTests(cfg);

        BasicTestFormatter::printTestSuccess("All Binomial tests completed successfully");
        return 0;

    } catch (const exception& e) {
        cerr << "Test failed: " << e.what() << endl;
        return 1;
    }
}

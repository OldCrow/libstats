// Basic test for GeometricDistribution — delegates to NegativeBinomial(r=1).
// Convention: X = number of failures before first success; support {0, 1, 2, …}.
#include "include/tests.h"
#include "include/basic_test_runner.h"
#include "libstats/distributions/geometric.h"

#include <cmath>
#include <iostream>
#include <random>

using namespace std;
using namespace stats;
using namespace stats::tests::fixtures;

int main() {
    BasicTestFormatter::printTestHeader("Geometric");

    // BasicDistConfig for Tests 6 and 8
    stats::tests::BasicDistConfig cfg{
        "Geometric",
        {0.0, 1.0, 2.0, 3.0, 4.0, 5.0},  // failure counts 0..5
        0.0, 19.5,                           // large batch: integers 0..19
        1e-12,                               // pdf_tolerance
        1e-12                                // cdf_tolerance (discrete floor property)
    };
    cfg.invalid_scenarios = {
        {"p = 0 (not in (0,1])",    [] { return GeometricDistribution::create(0.0).isError(); }},
        {"p < 0 (negative)",        [] { return GeometricDistribution::create(-0.1).isError(); }},
        {"p > 1 (above 1)",         [] { return GeometricDistribution::create(1.1).isError(); }},
        {"p = NaN",                 [] { return GeometricDistribution::create(
                                          std::numeric_limits<double>::quiet_NaN()).isError(); }},
        {"p = inf",                 [] { return GeometricDistribution::create(
                                          std::numeric_limits<double>::infinity()).isError(); }},
    };

    try {
        // Test 1: Constructors
        BasicTestFormatter::printTestStart(1, "Constructors and Destructor");
        cout << "Convention: X = failures before first success (support {0,1,2,...}).\n";
        cout << "Delegation: GeometricDistribution wraps NegativeBinomial(r=1, p).\n";

        auto default_geom = GeometricDistribution::create().value;
        BasicTestFormatter::printProperty("Default p", default_geom.getP());

        auto g05 = GeometricDistribution::create(0.5).value;
        auto g01 = GeometricDistribution::create(0.1).value;
        auto g10 = GeometricDistribution::create(1.0).value;

        BasicTestFormatter::printProperty("p=0.5 mean (expect 1.0)", g05.getMean());
        BasicTestFormatter::printProperty("p=0.1 mean (expect 9.0)", g01.getMean());
        BasicTestFormatter::printProperty("p=1.0 mean (expect 0.0)", g10.getMean());

        auto copy_g  = g05;
        auto move_g  = std::move(copy_g);
        BasicTestFormatter::printProperty("Copy/move p", move_g.getP());
        BasicTestFormatter::printTestSuccess("Constructors passed");
        BasicTestFormatter::printNewline();

        // Test 2: Parameter getters and setters
        BasicTestFormatter::printTestStart(2, "Parameter Getters and Setters");
        auto g = GeometricDistribution::create(0.3).value;
        BasicTestFormatter::printProperty("getP()", g.getP());
        BasicTestFormatter::printProperty("getPAtomic()", g.getPAtomic());

        g.setP(0.7);
        BasicTestFormatter::printProperty("After setP(0.7)", g.getP());

        auto r = g.trySetP(0.4);
        cout << "trySetP(0.4) ok: " << (r.isOk() ? "YES" : "NO") << endl;
        BasicTestFormatter::printProperty("After trySetP(0.4)", g.getP());

        auto r2 = g.trySetP(-1.0);
        cout << "trySetP(-1.0) isError: " << (r2.isError() ? "YES" : "NO") << endl;
        BasicTestFormatter::printProperty("p unchanged: ", g.getP());

        BasicTestFormatter::printTestSuccess("Getters and setters passed");
        BasicTestFormatter::printNewline();

        // Test 3: Core probability methods
        BasicTestFormatter::printTestStart(3, "Core Probability Methods");
        cout << "Geometric(p=0.5): PMF(k) = 0.5 * 0.5^k\n";
        auto g3 = GeometricDistribution::create(0.5).value;

        // PMF(0) = p = 0.5
        double pmf0 = g3.getProbability(0.0);
        cout << "PMF(0) = " << pmf0 << " [expect 0.5]\n";
        if (std::abs(pmf0 - 0.5) > 1e-12)
            throw std::runtime_error("PMF(0) accuracy failed");

        // PMF(1) = p*(1-p) = 0.25
        double pmf1 = g3.getProbability(1.0);
        cout << "PMF(1) = " << pmf1 << " [expect 0.25]\n";
        if (std::abs(pmf1 - 0.25) > 1e-12)
            throw std::runtime_error("PMF(1) accuracy failed");

        // LogPMF(0) = log(p) = log(0.5) ≈ -0.6931
        double lpmf0 = g3.getLogProbability(0.0);
        cout << "LogPMF(0) = " << lpmf0 << " [expect " << std::log(0.5) << "]\n";
        if (std::abs(lpmf0 - std::log(0.5)) > 1e-12)
            throw std::runtime_error("LogPMF(0) accuracy failed");

        // CDF(0) = P(X<=0) = P(X=0) = p = 0.5
        double cdf0 = g3.getCumulativeProbability(0.0);
        cout << "CDF(0) = " << cdf0 << " [expect 0.5]\n";
        if (std::abs(cdf0 - 0.5) > 1e-12)
            throw std::runtime_error("CDF(0) accuracy failed");

        // CDF(1) = 1 - (1-0.5)^2 = 0.75
        double cdf1 = g3.getCumulativeProbability(1.0);
        cout << "CDF(1) = " << cdf1 << " [expect 0.75]\n";
        if (std::abs(cdf1 - 0.75) > 1e-12)
            throw std::runtime_error("CDF(1) accuracy failed");

        // Moments
        cout << "Mean = " << g3.getMean() << " [expect 1.0]\n";
        cout << "Variance = " << g3.getVariance() << " [expect 2.0]\n";
        cout << "Skewness = " << g3.getSkewness() << " [expect ~2.828]\n";
        cout << "Kurtosis = " << g3.getKurtosis() << " [expect 10.0]\n";
        cout << "Entropy = " << g3.getEntropy() << " nats\n";
        cout << "Median = " << g3.getMedian() << " [expect 0 for p=0.5]\n";
        cout << "Mode = " << g3.getMode() << " [always 0]\n";

        if (std::abs(g3.getMean() - 1.0) > 1e-10) throw std::runtime_error("Mean failed");
        if (std::abs(g3.getVariance() - 2.0) > 1e-10) throw std::runtime_error("Variance failed");
        if (g3.getMode() != 0.0) throw std::runtime_error("Mode should be 0");
        if (g3.getMedian() != 0.0) throw std::runtime_error("Median failed for p=0.5");

        // Out-of-support: PMF(-1) = 0
        cout << "PMF(-1) = " << g3.getProbability(-1.0) << " [expect 0]\n";
        cout << "CDF(-1) = " << g3.getCumulativeProbability(-1.0) << " [expect 0]\n";
        cout << "isDiscrete: " << (g3.isDiscrete() ? "YES" : "NO") << "\n";
        cout << "Distribution name: " << g3.getDistributionName() << "\n";

        BasicTestFormatter::printTestSuccess("Core probability methods passed");
        BasicTestFormatter::printNewline();

        // Test 4: Random Sampling
        BasicTestFormatter::printTestStart(4, "Random Sampling");
        std::mt19937 rng(42);
        auto g4 = GeometricDistribution::create(0.3).value;  // mean = (1-0.3)/0.3 ≈ 2.333

        double s = g4.sample(rng);
        cout << "Single sample (expect >= 0): " << (s >= 0.0 ? "PASS" : "FAIL") << " (" << s << ")\n";
        if (s < 0.0) throw std::runtime_error("Sample out of support");

        auto samples = g4.sample(rng, 500);
        bool all_nonneg = true;
        double smean = 0.0;
        for (double sv : samples) {
            smean += sv;
            if (sv < 0.0) { all_nonneg = false; }
        }
        smean /= 500.0;
        cout << "All 500 samples >= 0: " << (all_nonneg ? "PASS" : "FAIL") << "\n";
        cout << "Sample mean (n=500, expect ~2.333): " << smean << "\n";
        if (!all_nonneg) throw std::runtime_error("Sample out of support");

        BasicTestFormatter::printTestSuccess("Sampling tests passed");
        BasicTestFormatter::printNewline();

        // Test 5: Distribution Management (fit, reset, toString)
        BasicTestFormatter::printTestStart(5, "Distribution Management");
        cout << "MLE: p_hat = 1/(1 + x_bar)  (closed form)\n";

        auto source = GeometricDistribution::create(0.4).value;  // true p=0.4
        auto fit_data = source.sample(rng, 300);
        auto g5 = GeometricDistribution::create().value;
        g5.fit(fit_data);
        cout << "Fitted p from Geo(0.4) data (expect ~0.4): " << g5.getP() << "\n";

        g5.reset();
        cout << "After reset p (expect 0.5): " << g5.getP() << "\n";
        cout << "toString: " << g5.toString() << "\n";
        if (std::abs(g5.getP() - 0.5) > 1e-10) throw std::runtime_error("Reset failed");

        BasicTestFormatter::printTestSuccess("Distribution management passed");
        BasicTestFormatter::printNewline();

        // Test 6: Batch + Test 7: Comparison/Stream
        auto g6 = GeometricDistribution::create(0.5).value;
        stats::tests::runBatchTests(cfg, g6);  // Test 6

        // Test 7: Comparison and Stream Operators
        BasicTestFormatter::printTestStart(7, "Comparison and Stream Operators");

        auto d1 = GeometricDistribution::create(0.3).value;
        auto d2 = GeometricDistribution::create(0.3).value;
        auto d3 = GeometricDistribution::create(0.7).value;
        cout << "d1 == d2 (p=0.3 vs p=0.3): " << (d1 == d2 ? "true" : "false") << "\n";
        cout << "d1 == d3 (p=0.3 vs p=0.7): " << (d1 == d3 ? "true" : "false") << "\n";
        cout << "d1 != d3: " << (d1 != d3 ? "true" : "false") << "\n";

        ostringstream oss;
        oss << d1;
        cout << "Stream output: " << oss.str() << "\n";
        istringstream iss(oss.str());
        auto parsed = GeometricDistribution::create().value;
        iss >> parsed;
        cout << "Stream round-trip p: " << parsed.getP() << "\n";
        if (std::abs(parsed.getP() - 0.3) > 1e-10) throw std::runtime_error("Stream round-trip failed");

        BasicTestFormatter::printTestSuccess("Comparison and stream tests passed");
        BasicTestFormatter::printNewline();

        // Test 8: Error Handling
        stats::tests::runErrorTests(cfg);

        BasicTestFormatter::printCompletionMessage("Geometric");
        BasicTestFormatter::printSummaryHeader();
        BasicTestFormatter::printSummaryItem("Delegation to NegativeBinomial(r=1)");
        BasicTestFormatter::printSummaryItem("PMF/LogPMF/CDF: p*(1-p)^k convention (failures before first success)");
        BasicTestFormatter::printSummaryItem("MLE: p_hat = 1/(1+x_bar) — closed form");
        BasicTestFormatter::printSummaryItem("Moments: mean=(1-p)/p, variance=(1-p)/p^2");
        BasicTestFormatter::printSummaryItem("Mode=0, Median=ceil(-ln2/ln(1-p))-1");
        BasicTestFormatter::printSummaryItem("SIMD/parallel batch via NegativeBinomial delegate");

        return 0;

    } catch (const exception& e) {
        cerr << "FATAL: " << e.what() << endl;
        return 1;
    }
}

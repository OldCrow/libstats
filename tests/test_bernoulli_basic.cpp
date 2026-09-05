// Basic test for BernoulliDistribution — delegates to Binomial(n=1, p).
// Support: x in {0, 1}.
#include "include/basic_test_runner.h"
#include "include/tests.h"
#include "libstats/distributions/bernoulli.h"

#include <cmath>
#include <iostream>
#include <random>

using namespace std;
using namespace stats;
using namespace stats::tests::fixtures;

int main() {
    BasicTestFormatter::printTestHeader("Bernoulli");

    // BasicDistConfig for Tests 6 and 8
    stats::tests::BasicDistConfig cfg{
        "Bernoulli", {0.0, 1.0, 0.0, 1.0, 1.0, 0.0},  // 0/1 outcomes
        0.0,         1.4,                             // large batch: values in {0,1}
        1e-12,                                        // pdf_tolerance
        1e-12                                         // cdf_tolerance
    };
    cfg.invalid_scenarios = {
        {"p < 0 (negative)", [] { return BernoulliDistribution::create(-0.1).isError(); }},
        {"p > 1 (above 1)", [] { return BernoulliDistribution::create(1.1).isError(); }},
        {"p = NaN",
         [] {
             return BernoulliDistribution::create(std::numeric_limits<double>::quiet_NaN())
                 .isError();
         }},
        {"p = inf",
         [] {
             return BernoulliDistribution::create(std::numeric_limits<double>::infinity())
                 .isError();
         }},
    };

    try {
        // Test 1: Constructors
        BasicTestFormatter::printTestStart(1, "Constructors and Destructor");
        cout << "Support: x in {0, 1}.\n";
        cout << "Delegation: BernoulliDistribution wraps Binomial(n=1, p).\n";

        auto default_bern = BernoulliDistribution::create().unwrap();
        BasicTestFormatter::printProperty("Default p", default_bern.getP());

        auto b05 = BernoulliDistribution::create(0.5).unwrap();
        auto b01 = BernoulliDistribution::create(0.1).unwrap();
        auto b00 = BernoulliDistribution::create(0.0).unwrap();
        auto b10 = BernoulliDistribution::create(1.0).unwrap();

        BasicTestFormatter::printProperty("p=0.5 mean (expect 0.5)", b05.getMean());
        BasicTestFormatter::printProperty("p=0.1 mean (expect 0.1)", b01.getMean());
        BasicTestFormatter::printProperty("p=0.0 mean (expect 0.0)", b00.getMean());
        BasicTestFormatter::printProperty("p=1.0 mean (expect 1.0)", b10.getMean());

        auto copy_b = b05;
        auto move_b = std::move(copy_b);
        BasicTestFormatter::printProperty("Copy/move p", move_b.getP());
        BasicTestFormatter::printTestSuccess("Constructors passed");
        BasicTestFormatter::printNewline();

        // Test 2: Parameter getters and setters
        BasicTestFormatter::printTestStart(2, "Parameter Getters and Setters");
        auto b = BernoulliDistribution::create(0.3).unwrap();
        BasicTestFormatter::printProperty("getP()", b.getP());
        BasicTestFormatter::printProperty("getPAtomic()", b.getPAtomic());

        b.setP(0.7);
        BasicTestFormatter::printProperty("After setP(0.7)", b.getP());

        auto r = b.trySetP(0.4);
        cout << "trySetP(0.4) ok: " << (r.isOk() ? "YES" : "NO") << endl;
        BasicTestFormatter::printProperty("After trySetP(0.4)", b.getP());

        auto r2 = b.trySetP(-1.0);
        cout << "trySetP(-1.0) isError: " << (r2.isError() ? "YES" : "NO") << endl;
        BasicTestFormatter::printProperty("p unchanged: ", b.getP());

        BasicTestFormatter::printTestSuccess("Getters and setters passed");
        BasicTestFormatter::printNewline();

        // Test 3: Core probability methods
        BasicTestFormatter::printTestStart(3, "Core Probability Methods");
        cout << "Bernoulli(p=0.5): PMF(0)=0.5, PMF(1)=0.5\n";
        auto b3 = BernoulliDistribution::create(0.5).unwrap();

        double pmf0 = b3.getProbability(0.0);
        cout << "PMF(0) = " << pmf0 << " [expect 0.5]\n";
        if (std::abs(pmf0 - 0.5) > 1e-12)
            throw std::runtime_error("PMF(0) accuracy failed");

        double pmf1 = b3.getProbability(1.0);
        cout << "PMF(1) = " << pmf1 << " [expect 0.5]\n";
        if (std::abs(pmf1 - 0.5) > 1e-12)
            throw std::runtime_error("PMF(1) accuracy failed");

        double lpmf1 = b3.getLogProbability(1.0);
        cout << "LogPMF(1) = " << lpmf1 << " [expect " << std::log(0.5) << "]\n";
        if (std::abs(lpmf1 - std::log(0.5)) > 1e-12)
            throw std::runtime_error("LogPMF(1) accuracy failed");

        double cdf0 = b3.getCumulativeProbability(0.0);
        cout << "CDF(0) = " << cdf0 << " [expect 0.5]\n";
        if (std::abs(cdf0 - 0.5) > 1e-12)
            throw std::runtime_error("CDF(0) accuracy failed");

        double cdf1 = b3.getCumulativeProbability(1.0);
        cout << "CDF(1) = " << cdf1 << " [expect 1.0]\n";
        if (std::abs(cdf1 - 1.0) > 1e-12)
            throw std::runtime_error("CDF(1) accuracy failed");

        // Skewed distribution
        auto b3_skew = BernoulliDistribution::create(0.2).unwrap();
        cout << "Mean (p=0.2) = " << b3_skew.getMean() << " [expect 0.2]\n";
        cout << "Variance (p=0.2) = " << b3_skew.getVariance() << " [expect 0.16]\n";
        if (std::abs(b3_skew.getMean() - 0.2) > 1e-10)
            throw std::runtime_error("Mean failed");
        if (std::abs(b3_skew.getVariance() - 0.16) > 1e-10)
            throw std::runtime_error("Variance failed");

        cout << "Mode (p=0.2, expect 0) = " << b3_skew.getMode() << "\n";
        cout << "Median (p=0.2, expect 0) = " << b3_skew.getMedian() << "\n";
        cout << "Entropy (p=0.5) = " << b3.getEntropy() << " nats\n";

        // Out-of-support
        cout << "PMF(2) = " << b3.getProbability(2.0) << " [expect 0]\n";
        cout << "PMF(-1) = " << b3.getProbability(-1.0) << " [expect 0]\n";
        cout << "isDiscrete: " << (b3.isDiscrete() ? "YES" : "NO") << "\n";
        cout << "Distribution name: " << b3.getDistributionName() << "\n";

        BasicTestFormatter::printTestSuccess("Core probability methods passed");
        BasicTestFormatter::printNewline();

        // Test 4: Random Sampling
        BasicTestFormatter::printTestStart(4, "Random Sampling");
        std::mt19937 rng(42);
        auto b4 = BernoulliDistribution::create(0.3).unwrap();

        double s = b4.sample(rng);
        cout << "Single sample (expect 0 or 1): " << ((s == 0.0 || s == 1.0) ? "PASS" : "FAIL")
             << " (" << s << ")\n";
        if (s != 0.0 && s != 1.0)
            throw std::runtime_error("Sample out of support");

        auto samples = b4.sample(rng, 500);
        bool all_valid = true;
        double smean = 0.0;
        for (double sv : samples) {
            smean += sv;
            if (sv != 0.0 && sv != 1.0) {
                all_valid = false;
            }
        }
        smean /= 500.0;
        cout << "All 500 samples in {0,1}: " << (all_valid ? "PASS" : "FAIL") << "\n";
        cout << "Sample mean (n=500, expect ~0.3): " << smean << "\n";
        if (!all_valid)
            throw std::runtime_error("Sample out of support");

        BasicTestFormatter::printTestSuccess("Sampling tests passed");
        BasicTestFormatter::printNewline();

        // Test 5: Distribution Management (fit, reset, toString)
        BasicTestFormatter::printTestStart(5, "Distribution Management");
        cout << "MLE: p_hat = x_bar (closed form)\n";

        auto source = BernoulliDistribution::create(0.4).unwrap();  // true p=0.4
        auto fit_data = source.sample(rng, 500);
        auto b5 = BernoulliDistribution::create().unwrap();
        b5.fit(fit_data);
        cout << "Fitted p from Bernoulli(0.4) data (expect ~0.4): " << b5.getP() << "\n";

        b5.reset();
        cout << "After reset p (expect 0.5): " << b5.getP() << "\n";
        cout << "toString: " << b5.toString() << "\n";
        if (std::abs(b5.getP() - 0.5) > 1e-10)
            throw std::runtime_error("Reset failed");

        BasicTestFormatter::printTestSuccess("Distribution management passed");
        BasicTestFormatter::printNewline();

        // Test 6: Batch + Test 7: Comparison/Stream
        auto b6 = BernoulliDistribution::create(0.5).unwrap();
        stats::tests::runBatchTests(cfg, b6);  // Test 6

        // Test 7: Comparison and Stream Operators
        BasicTestFormatter::printTestStart(7, "Comparison and Stream Operators");

        auto d1 = BernoulliDistribution::create(0.3).unwrap();
        auto d2 = BernoulliDistribution::create(0.3).unwrap();
        auto d3 = BernoulliDistribution::create(0.7).unwrap();
        cout << "d1 == d2 (p=0.3 vs p=0.3): " << (d1 == d2 ? "true" : "false") << "\n";
        cout << "d1 == d3 (p=0.3 vs p=0.7): " << (d1 == d3 ? "true" : "false") << "\n";
        cout << "d1 != d3: " << (d1 != d3 ? "true" : "false") << "\n";

        ostringstream oss;
        oss << d1;
        cout << "Stream output: " << oss.str() << "\n";
        istringstream iss(oss.str());
        auto parsed = BernoulliDistribution::create().unwrap();
        iss >> parsed;
        cout << "Stream round-trip p: " << parsed.getP() << "\n";
        if (std::abs(parsed.getP() - 0.3) > 1e-10)
            throw std::runtime_error("Stream round-trip failed");

        BasicTestFormatter::printTestSuccess("Comparison and stream tests passed");
        BasicTestFormatter::printNewline();

        // Test 8: Error Handling
        stats::tests::runErrorTests(cfg);

        BasicTestFormatter::printCompletionMessage("Bernoulli");
        BasicTestFormatter::printSummaryHeader();
        BasicTestFormatter::printSummaryItem("Delegation to Binomial(n=1)");
        BasicTestFormatter::printSummaryItem("PMF: P(1)=p, P(0)=1-p");
        BasicTestFormatter::printSummaryItem("MLE: p_hat = x_bar — closed form");
        BasicTestFormatter::printSummaryItem("Moments: mean=p, variance=p(1-p)");
        BasicTestFormatter::printSummaryItem("Quantile: right-continuous discrete inverse (#104)");
        BasicTestFormatter::printSummaryItem("SIMD/parallel batch via Binomial delegate");

        return 0;

    } catch (const exception& e) {
        cerr << "FATAL: " << e.what() << endl;
        return 1;
    }
}

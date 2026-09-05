// Basic test for GumbelDistribution (Type I extreme value, MAX-stable / gumbel_r).
// PDF: (1/beta)*exp(-z - e^-z); CDF: exp(-e^-z); support: all reals.
// The min-stable variant (gumbel_l) is deliberately absent — issue #54 kickoff
// decision, 2026-09-02: it is the law of -X, a user-side sign flip.
// Reference values quoted below come from mpmath (mp.dps = 60, 17 sig digits).
#include "include/basic_test_runner.h"
#include "include/tests.h"
#include "libstats/distributions/gumbel.h"

#include <cmath>
#include <iostream>
#include <random>

using namespace std;
using namespace stats;
using namespace stats::tests::fixtures;

int main() {
    BasicTestFormatter::printTestHeader("Gumbel");

    stats::tests::BasicDistConfig cfg{
        "Gumbel", {-2.0, -0.5, 0.0, 1.0, 3.0}, -3.0, 6.0,
        1e-10,  // pdf_tolerance
        1e-10   // cdf_tolerance
    };
    cfg.invalid_scenarios = {
        {"beta = 0", [] { return GumbelDistribution::create(0.0, 0.0).isError(); }},
        {"beta < 0", [] { return GumbelDistribution::create(0.0, -1.0).isError(); }},
        {"mu = inf",
         [] {
             return GumbelDistribution::create(std::numeric_limits<double>::infinity(), 1.0)
                 .isError();
         }},
        {"beta = NaN",
         [] {
             return GumbelDistribution::create(0.0, std::numeric_limits<double>::quiet_NaN())
                 .isError();
         }},
    };

    try {
        // Test 1: Constructors
        BasicTestFormatter::printTestStart(1, "Constructors and Destructor");
        cout << "Gumbel(mu, beta): max-stable extreme-value law, right-skewed.\n";

        auto def = GumbelDistribution::create().unwrap();
        BasicTestFormatter::printProperty("Default mu   (expect 0)", def.getMu());
        BasicTestFormatter::printProperty("Default beta (expect 1)", def.getBeta());

        auto std_g = GumbelDistribution::create(0.0, 1.0).unwrap();
        auto g_5_2 = GumbelDistribution::create(5.0, 2.0).unwrap();
        // mean = mu + gamma*beta = 5 + 2*0.5772156649015329 = 6.1544313298030658
        BasicTestFormatter::printProperty("G(5,2) mean (expect 6.1544313298)", g_5_2.getMean());
        BasicTestFormatter::printProperty("G(5,2) isStandard (expect 0)", g_5_2.isStandard());
        BasicTestFormatter::printProperty("G(0,1) isStandard (expect 1)", std_g.isStandard());

        auto copy_g = std_g;
        auto move_g = std::move(copy_g);
        BasicTestFormatter::printProperty("Copy/move mu", move_g.getMu());
        BasicTestFormatter::printTestSuccess("Constructors passed");
        BasicTestFormatter::printNewline();

        // Test 2: Parameter getters and setters
        BasicTestFormatter::printTestStart(2, "Parameter Getters and Setters");
        auto g = GumbelDistribution::create(1.0, 2.0).unwrap();
        BasicTestFormatter::printProperty("getMu()", g.getMu());
        BasicTestFormatter::printProperty("getBeta()", g.getBeta());
        BasicTestFormatter::printProperty("getMuAtomic()", g.getMuAtomic());
        BasicTestFormatter::printProperty("getBetaAtomic()", g.getBetaAtomic());

        g.setMu(-1.0);
        BasicTestFormatter::printProperty("After setMu(-1)", g.getMu());
        g.setBeta(0.5);
        BasicTestFormatter::printProperty("After setBeta(0.5)", g.getBeta());
        g.setParameters(3.0, 1.0);
        BasicTestFormatter::printProperty("After setParameters(3,1) mu", g.getMu());

        auto r1 = g.trySetMu(0.0);
        cout << "trySetMu(0) ok: " << (r1.isOk() ? "YES" : "NO") << "\n";
        auto r2 = g.trySetBeta(-1.0);
        cout << "trySetBeta(-1) isError: " << (r2.isError() ? "YES" : "NO") << "\n";

        BasicTestFormatter::printTestSuccess("Getters/setters passed");
        BasicTestFormatter::printNewline();

        // Test 3: Core probability methods
        BasicTestFormatter::printTestStart(3, "Core Probability Methods");
        cout << "Standard Gumbel(mu=0, beta=1):\n";
        cout << "  PDF(0) = CDF(0) = 1/e = 0.36787944117144232 (both, at z = 0)\n";
        auto sg = GumbelDistribution::create(0.0, 1.0).unwrap();

        double pdf0 = sg.getProbability(0.0);
        cout << "PDF(0) = " << pdf0 << " [expect 0.36787944117144232]\n";
        if (std::abs(pdf0 - 0.36787944117144232) > 1e-15)
            throw runtime_error("PDF(0) failed");

        double lp0 = sg.getLogProbability(0.0);
        cout << "LogPDF(0) = " << lp0 << " [expect -1.0 exactly]\n";
        if (std::abs(lp0 + 1.0) > 1e-15)
            throw runtime_error("LogPDF(0) failed");

        double cdf0 = sg.getCumulativeProbability(0.0);
        cout << "CDF(0) = " << cdf0 << " [expect 0.36787944117144232]\n";
        if (std::abs(cdf0 - 0.36787944117144232) > 1e-15)
            throw runtime_error("CDF(0) failed");

        // mpmath: CDF(1; 0, 1) = 0.69220062755534635, PDF(1) = 0.25464638004358250
        if (std::abs(sg.getCumulativeProbability(1.0) - 0.69220062755534635) > 1e-14)
            throw runtime_error("CDF(1) failed");
        if (std::abs(sg.getProbability(1.0) - 0.25464638004358250) > 1e-15)
            throw runtime_error("PDF(1) failed");

        // Deep lower tail: the double exponential must land on exactly 0, not on
        // a clamped near-zero. CDF(-10) has true value ~1e-9566.
        double cdf_m10 = sg.getCumulativeProbability(-10.0);
        cout << "CDF(-10) = " << cdf_m10 << " [expect exactly 0]\n";
        if (cdf_m10 != 0.0)
            throw runtime_error("CDF(-10) did not underflow to exactly 0");
        // ...while the LogPDF there is still an ordinary finite number:
        // -(-10) - e^10 = -22016.465794806717 (mpmath)
        double lp_m10 = sg.getLogProbability(-10.0);
        cout << "LogPDF(-10) = " << lp_m10 << " [expect -22016.465794806717]\n";
        if (std::abs(lp_m10 - (-22016.465794806717)) > 1e-9)
            throw runtime_error("LogPDF(-10) failed");
        if (sg.getProbability(-10.0) != 0.0)
            throw runtime_error("PDF(-10) did not underflow to exactly 0");

        // Deep upper tail: CDF must reach exactly 1.
        double cdf_p40 = sg.getCumulativeProbability(40.0);
        cout << "CDF(40) = " << cdf_p40 << " [expect exactly 1]\n";
        if (cdf_p40 != 1.0)
            throw runtime_error("CDF(40) did not saturate to exactly 1");

        // Quantile round-trip
        for (double p : {1e-12, 0.25, 0.5, 0.75, 1.0 - 1e-12}) {
            const double q = sg.getQuantile(p);
            const double back = sg.getCumulativeProbability(q);
            cout << "Q(" << p << ") = " << q << "  CDF(Q) = " << back << "\n";
            if (std::abs(back - p) > 1e-9 * std::max(p, 1e-12))
                throw runtime_error("Quantile round-trip failed");
        }

        // Moments (mpmath, Gumbel(0,1))
        cout << "Mean     = " << sg.getMean() << " [expect 0.57721566490153286]\n";
        cout << "Variance = " << sg.getVariance() << " [expect 1.6449340668482264]\n";
        cout << "Skewness = " << sg.getSkewness() << " [expect 1.1395470994046487]\n";
        cout << "Kurtosis = " << sg.getKurtosis() << " [expect 2.4]\n";
        cout << "Entropy  = " << sg.getEntropy() << " [expect 1.5772156649015329]\n";
        cout << "Median   = " << sg.getMedian() << " [expect 0.36651292058166433]\n";
        cout << "Mode     = " << sg.getMode() << " [expect 0.0]\n";

        if (std::abs(sg.getMean() - 0.57721566490153286) > 1e-15)
            throw runtime_error("Mean failed");
        if (std::abs(sg.getVariance() - 1.6449340668482264) > 1e-14)
            throw runtime_error("Variance failed");
        if (std::abs(sg.getSkewness() - 1.1395470994046487) > 1e-15)
            throw runtime_error("Skewness failed");
        if (std::abs(sg.getKurtosis() - 2.4) > 1e-15)
            throw runtime_error("Kurtosis failed");
        if (std::abs(sg.getEntropy() - 1.5772156649015329) > 1e-14)
            throw runtime_error("Entropy failed");
        if (std::abs(sg.getMedian() - 0.36651292058166433) > 1e-15)
            throw runtime_error("Median failed");
        // Median must be the 0.5 quantile
        if (std::abs(sg.getMedian() - sg.getQuantile(0.5)) > 1e-15)
            throw runtime_error("Median != Quantile(0.5)");

        cout << "isDiscrete: " << (sg.isDiscrete() ? "YES" : "NO") << "\n";
        BasicTestFormatter::printTestSuccess("Core probability methods passed");
        BasicTestFormatter::printNewline();

        // Test 4: Random Sampling
        BasicTestFormatter::printTestStart(4, "Random Sampling");
        mt19937 rng(42);
        auto g4 = GumbelDistribution::create(3.0, 2.0).unwrap();  // mean = 3 + 2*gamma

        double s = g4.sample(rng);
        cout << "Single sample: " << s << "\n";
        if (!std::isfinite(s))
            throw runtime_error("Sample not finite");

        auto samples = g4.sample(rng, 500);
        double smean = 0.0;
        for (double sv : samples)
            smean += sv;
        smean /= 500.0;
        cout << "Sample mean (n=500, expect ~" << (3.0 + 2.0 * 0.5772156649015329)
             << "): " << smean << "\n";

        BasicTestFormatter::printTestSuccess("Sampling tests passed");
        BasicTestFormatter::printNewline();

        // Test 5: Distribution Management
        BasicTestFormatter::printTestStart(5, "Distribution Management");
        cout << "Fit: METHOD OF MOMENTS (not MLE) — beta_hat = sd*sqrt(6)/pi, "
                "mu_hat = mean - gamma*beta_hat\n";

        auto source = GumbelDistribution::create(2.0, 0.5).unwrap();
        auto fit_data = source.sample(rng, 2000);
        auto g5 = GumbelDistribution::create().unwrap();
        g5.fit(fit_data);
        cout << "Fitted mu   (from Gumbel(2, 0.5), expect ~2):   " << g5.getMu() << "\n";
        cout << "Fitted beta (from Gumbel(2, 0.5), expect ~0.5): " << g5.getBeta() << "\n";
        if (std::abs(g5.getMu() - 2.0) > 0.1 || std::abs(g5.getBeta() - 0.5) > 0.1)
            throw runtime_error("Moment fit did not recover the source parameters");

        g5.reset();
        if (std::abs(g5.getMu()) > 1e-10 || std::abs(g5.getBeta() - 1.0) > 1e-10)
            throw runtime_error("Reset failed");
        cout << "After reset: mu=0, beta=1 (PASS)\n";
        cout << "toString: " << g5.toString() << "\n";

        BasicTestFormatter::printTestSuccess("Distribution management passed");
        BasicTestFormatter::printNewline();

        // Tests 6 and 8
        auto g6 = GumbelDistribution::create(0.0, 1.0).unwrap();
        stats::tests::runBatchTests(cfg, g6);

        // Test 7: Comparison and Stream Operators
        BasicTestFormatter::printTestStart(7, "Comparison and Stream Operators");
        auto d1 = GumbelDistribution::create(0.0, 1.0).unwrap();
        auto d2 = GumbelDistribution::create(0.0, 1.0).unwrap();
        auto d3 = GumbelDistribution::create(1.0, 2.0).unwrap();
        cout << "d1 == d2: " << (d1 == d2 ? "true" : "false") << "\n";
        cout << "d1 != d3: " << (d1 != d3 ? "true" : "false") << "\n";
        if (!(d1 == d2) || !(d1 != d3))
            throw runtime_error("Comparison operators failed");

        ostringstream oss;
        oss << d1;
        cout << "Stream: " << oss.str() << "\n";
        istringstream iss(oss.str());
        auto parsed = GumbelDistribution::create(7.0, 9.0).unwrap();
        iss >> parsed;
        if (std::abs(parsed.getMu()) > 1e-10 || std::abs(parsed.getBeta() - 1.0) > 1e-10)
            throw runtime_error("Stream round-trip failed");
        cout << "Stream round-trip: mu=" << parsed.getMu() << " beta=" << parsed.getBeta() << "\n";

        BasicTestFormatter::printTestSuccess("Comparison and stream passed");
        BasicTestFormatter::printNewline();

        stats::tests::runErrorTests(cfg);

        BasicTestFormatter::printCompletionMessage("Gumbel");
        BasicTestFormatter::printSummaryHeader();
        BasicTestFormatter::printSummaryItem(
            "Max-stable (gumbel_r) only; gumbel_l deferred per issue #54 kickoff");
        BasicTestFormatter::printSummaryItem(
            "LogPDF exact in log space: -log(beta) - z - e^-z");
        BasicTestFormatter::printSummaryItem(
            "CDF exp(-exp(-z)) reaches exactly 0 and 1 at the extremes");
        BasicTestFormatter::printSummaryItem(
            "Moments: mean=mu+gamma*beta, var=pi^2*beta^2/6, skewness=1.13955, kurtosis=12/5");
        BasicTestFormatter::printSummaryItem(
            "Fit is method-of-moments, NOT the iterative MLE (documented in gumbel.h)");

        return 0;
    } catch (const exception& e) {
        cerr << "FATAL: " << e.what() << endl;
        return 1;
    }
}

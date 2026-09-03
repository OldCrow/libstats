// Basic test for LogisticDistribution (sigmoid CDF, standalone).
// PDF: e^(-|z|)/(s*(1+e^(-|z|))^2); CDF: sign-branched sigmoid; support: all reals.
// Reference values quoted below come from mpmath (mp.dps = 60, printed to 17
// significant digits) — see the header block of test_logistic_enhanced.cpp.
#include "include/basic_test_runner.h"
#include "include/tests.h"
#include "libstats/distributions/logistic.h"

#include <cmath>
#include <iostream>
#include <random>

using namespace std;
using namespace stats;
using namespace stats::tests::fixtures;

int main() {
    BasicTestFormatter::printTestHeader("Logistic");

    stats::tests::BasicDistConfig cfg{
        "Logistic", {-3.0, -1.0, 0.0, 1.0, 3.0}, -5.0, 5.0,
        1e-10,  // pdf_tolerance (vector path uses log(1+e) vs scalar log1p(e): ~2e-16)
        1e-10   // cdf_tolerance
    };
    cfg.invalid_scenarios = {
        {"s = 0", [] { return LogisticDistribution::create(0.0, 0.0).isError(); }},
        {"s < 0", [] { return LogisticDistribution::create(0.0, -1.0).isError(); }},
        {"mu = inf",
         [] {
             return LogisticDistribution::create(std::numeric_limits<double>::infinity(), 1.0)
                 .isError();
         }},
        {"s = NaN",
         [] {
             return LogisticDistribution::create(0.0, std::numeric_limits<double>::quiet_NaN())
                 .isError();
         }},
    };

    try {
        // Test 1: Constructors
        BasicTestFormatter::printTestStart(1, "Constructors and Destructor");
        cout << "Logistic(mu, s): symmetric about mu, CDF is the standard sigmoid.\n";

        auto def = LogisticDistribution::create().unwrap();
        BasicTestFormatter::printProperty("Default mu (expect 0)", def.getMu());
        BasicTestFormatter::printProperty("Default s  (expect 1)", def.getS());

        auto std_log = LogisticDistribution::create(0.0, 1.0).unwrap();
        auto log_5_2 = LogisticDistribution::create(5.0, 2.0).unwrap();
        BasicTestFormatter::printProperty("Log(5,2) mean (expect 5)", log_5_2.getMean());
        BasicTestFormatter::printProperty("Log(5,2) isStandard (expect 0)", log_5_2.isStandard());
        BasicTestFormatter::printProperty("Log(0,1) isStandard (expect 1)", std_log.isStandard());

        auto copy_l = std_log;
        auto move_l = std::move(copy_l);
        BasicTestFormatter::printProperty("Copy/move mu", move_l.getMu());
        BasicTestFormatter::printTestSuccess("Constructors passed");
        BasicTestFormatter::printNewline();

        // Test 2: Parameter getters and setters
        BasicTestFormatter::printTestStart(2, "Parameter Getters and Setters");
        auto l = LogisticDistribution::create(1.0, 2.0).unwrap();
        BasicTestFormatter::printProperty("getMu()", l.getMu());
        BasicTestFormatter::printProperty("getS()", l.getS());
        BasicTestFormatter::printProperty("getMuAtomic()", l.getMuAtomic());
        BasicTestFormatter::printProperty("getSAtomic()", l.getSAtomic());

        l.setMu(-1.0);
        BasicTestFormatter::printProperty("After setMu(-1)", l.getMu());
        l.setS(0.5);
        BasicTestFormatter::printProperty("After setS(0.5)", l.getS());
        l.setParameters(3.0, 1.0);
        BasicTestFormatter::printProperty("After setParameters(3,1) mu", l.getMu());

        auto r1 = l.trySetMu(0.0);
        cout << "trySetMu(0) ok: " << (r1.isOk() ? "YES" : "NO") << "\n";
        auto r2 = l.trySetS(-1.0);
        cout << "trySetS(-1) isError: " << (r2.isError() ? "YES" : "NO") << "\n";

        BasicTestFormatter::printTestSuccess("Getters/setters passed");
        BasicTestFormatter::printNewline();

        // Test 3: Core probability methods
        BasicTestFormatter::printTestStart(3, "Core Probability Methods");
        cout << "Standard Logistic(mu=0, s=1):\n";
        cout << "  PDF(0) = 1/4 = 0.25; LogPDF(0) = -log(4) = -1.3862943611198906\n";
        cout << "  CDF(0) = 0.5 (exactly, by symmetry)\n";
        auto sl = LogisticDistribution::create(0.0, 1.0).unwrap();

        double pdf0 = sl.getProbability(0.0);
        cout << "PDF(0) = " << pdf0 << " [expect 0.25]\n";
        if (std::abs(pdf0 - 0.25) > 1e-15)
            throw runtime_error("PDF(0) failed");

        double lp0 = sl.getLogProbability(0.0);
        cout << "LogPDF(0) = " << lp0 << " [expect -1.3862943611198906]\n";
        if (std::abs(lp0 - (-1.3862943611198906)) > 1e-14)
            throw runtime_error("LogPDF(0) failed");

        double cdf0 = sl.getCumulativeProbability(0.0);
        cout << "CDF(0) = " << cdf0 << " [expect 0.5]\n";
        if (std::abs(cdf0 - 0.5) > 1e-15)
            throw runtime_error("CDF(0) failed");

        // mpmath: CDF(2; 0, 1) = 0.88079707797788244
        double cdf2 = sl.getCumulativeProbability(2.0);
        cout << "CDF(2) = " << cdf2 << " [expect 0.88079707797788244]\n";
        if (std::abs(cdf2 - 0.88079707797788244) > 1e-14)
            throw runtime_error("CDF(2) failed");

        // Far lower tail: CDF(-40) = 4.2483542552915890e-18 (mpmath). The naive
        // 1/(1+e^(-z)) form would evaluate e^(+40) here; the stable form does not.
        double cdf_m40 = sl.getCumulativeProbability(-40.0);
        cout << "CDF(-40) = " << cdf_m40 << " [expect 4.2483542552915890e-18]\n";
        if (std::abs(cdf_m40 - 4.2483542552915890e-18) > 1e-32)
            throw runtime_error("CDF(-40) lower-tail failed");

        // Quantile round-trip, including the p -> 1 tail that a naive
        // log(p/(1-p)) would destroy.
        for (double p : {1e-12, 0.25, 0.5, 0.75, 1.0 - 1e-12}) {
            const double q = sl.getQuantile(p);
            const double back = sl.getCumulativeProbability(q);
            cout << "Q(" << p << ") = " << q << "  CDF(Q) = " << back << "\n";
            if (std::abs(back - p) > 1e-9 * std::max(p, 1e-12))
                throw runtime_error("Quantile round-trip failed");
        }

        // Moments (mpmath: variance = pi^2/3 = 3.2898681336964529, entropy = 2)
        cout << "Mean = " << sl.getMean() << " [expect 0.0]\n";
        cout << "Variance = " << sl.getVariance() << " [expect 3.2898681336964529]\n";
        cout << "Skewness = " << sl.getSkewness() << " [expect 0.0]\n";
        cout << "Kurtosis = " << sl.getKurtosis() << " [expect 1.2]\n";
        cout << "Entropy  = " << sl.getEntropy() << " [expect 2.0]\n";
        cout << "Median   = " << sl.getMedian() << " [expect 0.0]\n";
        cout << "Mode     = " << sl.getMode() << " [expect 0.0]\n";

        if (std::abs(sl.getMean()) > 1e-15)
            throw runtime_error("Mean failed");
        if (std::abs(sl.getVariance() - 3.2898681336964529) > 1e-14)
            throw runtime_error("Variance failed");
        if (std::abs(sl.getSkewness()) > 1e-15)
            throw runtime_error("Skewness failed");
        if (std::abs(sl.getKurtosis() - 1.2) > 1e-15)
            throw runtime_error("Kurtosis failed");
        if (std::abs(sl.getEntropy() - 2.0) > 1e-14)
            throw runtime_error("Entropy failed");

        // Symmetry: PDF(mu+d) == PDF(mu-d) for any d
        auto lg = LogisticDistribution::create(2.0, 1.5).unwrap();
        for (double d : {0.5, 1.0, 2.5}) {
            double lo = lg.getProbability(2.0 - d);
            double hi = lg.getProbability(2.0 + d);
            if (std::abs(lo - hi) > 1e-15)
                throw runtime_error("Symmetry violated");
        }
        cout << "Symmetry check: PASS\n";
        cout << "isDiscrete: " << (sl.isDiscrete() ? "YES" : "NO") << "\n";

        BasicTestFormatter::printTestSuccess("Core probability methods passed");
        BasicTestFormatter::printNewline();

        // Test 4: Random Sampling
        BasicTestFormatter::printTestStart(4, "Random Sampling");
        mt19937 rng(42);
        auto lg4 = LogisticDistribution::create(3.0, 2.0).unwrap();  // mean = 3

        double s = lg4.sample(rng);
        cout << "Single sample: " << s << "\n";
        if (!std::isfinite(s))
            throw runtime_error("Sample not finite");

        auto samples = lg4.sample(rng, 500);
        double smean = 0.0;
        for (double sv : samples)
            smean += sv;
        smean /= 500.0;
        cout << "Sample mean (n=500, expect ~3.0): " << smean << "\n";

        BasicTestFormatter::printTestSuccess("Sampling tests passed");
        BasicTestFormatter::printNewline();

        // Test 5: Distribution Management
        BasicTestFormatter::printTestStart(5, "Distribution Management");
        cout << "Conditional MLE: mu_hat = median, s_hat solves sum z*tanh(z/2) = n\n";

        auto source = LogisticDistribution::create(2.0, 0.5).unwrap();
        auto fit_data = source.sample(rng, 2000);
        auto l5 = LogisticDistribution::create().unwrap();
        l5.fit(fit_data);
        cout << "Fitted mu (from Logistic(2, 0.5), expect ~2): " << l5.getMu() << "\n";
        cout << "Fitted s  (from Logistic(2, 0.5), expect ~0.5): " << l5.getS() << "\n";
        if (std::abs(l5.getMu() - 2.0) > 0.1 || std::abs(l5.getS() - 0.5) > 0.1)
            throw runtime_error("MLE fit did not recover the source parameters");

        l5.reset();
        if (std::abs(l5.getMu()) > 1e-10 || std::abs(l5.getS() - 1.0) > 1e-10)
            throw runtime_error("Reset failed");
        cout << "After reset: mu=0, s=1 (PASS)\n";
        cout << "toString: " << l5.toString() << "\n";

        BasicTestFormatter::printTestSuccess("Distribution management passed");
        BasicTestFormatter::printNewline();

        // Tests 6 and 8
        auto l6 = LogisticDistribution::create(0.0, 1.0).unwrap();
        stats::tests::runBatchTests(cfg, l6);

        // Test 7: Comparison and Stream Operators
        BasicTestFormatter::printTestStart(7, "Comparison and Stream Operators");
        auto d1 = LogisticDistribution::create(0.0, 1.0).unwrap();
        auto d2 = LogisticDistribution::create(0.0, 1.0).unwrap();
        auto d3 = LogisticDistribution::create(1.0, 2.0).unwrap();
        cout << "d1 == d2: " << (d1 == d2 ? "true" : "false") << "\n";
        cout << "d1 != d3: " << (d1 != d3 ? "true" : "false") << "\n";
        if (!(d1 == d2) || !(d1 != d3))
            throw runtime_error("Comparison operators failed");

        ostringstream oss;
        oss << d1;
        cout << "Stream: " << oss.str() << "\n";
        istringstream iss(oss.str());
        auto parsed = LogisticDistribution::create(7.0, 9.0).unwrap();
        iss >> parsed;
        if (std::abs(parsed.getMu()) > 1e-10 || std::abs(parsed.getS() - 1.0) > 1e-10)
            throw runtime_error("Stream round-trip failed");
        cout << "Stream round-trip: mu=" << parsed.getMu() << " s=" << parsed.getS() << "\n";

        BasicTestFormatter::printTestSuccess("Comparison and stream passed");
        BasicTestFormatter::printNewline();

        stats::tests::runErrorTests(cfg);

        BasicTestFormatter::printCompletionMessage("Logistic");
        BasicTestFormatter::printSummaryHeader();
        BasicTestFormatter::printSummaryItem(
            "Standalone implementation: symmetric vector_exp/vector_log pipeline");
        BasicTestFormatter::printSummaryItem(
            "Only -|z| is ever exponentiated, so neither tail can overflow");
        BasicTestFormatter::printSummaryItem(
            "Quantile keeps log(p) - log1p(-p) split (naive logit loses p -> 1)");
        BasicTestFormatter::printSummaryItem(
            "Moments: mean=mode=median=mu, variance=s^2*pi^2/3, skewness=0, kurtosis=6/5");
        BasicTestFormatter::printSummaryItem(
            "Fit: mu_hat=median (closed form), s_hat by safeguarded Newton on log s");

        return 0;
    } catch (const exception& e) {
        cerr << "FATAL: " << e.what() << endl;
        return 1;
    }
}

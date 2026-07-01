// Basic test for LaplaceDistribution (double-exponential, standalone).
// PDF: (1/2b)*exp(-|x-mu|/b); MLE: mu=median, b=MAD; support: all reals.
#include "include/tests.h"
#include "include/basic_test_runner.h"
#include "libstats/distributions/laplace.h"

#include <cmath>
#include <iostream>
#include <random>

using namespace std;
using namespace stats;
using namespace stats::tests::fixtures;

int main() {
    BasicTestFormatter::printTestHeader("Laplace");

    stats::tests::BasicDistConfig cfg{
        "Laplace",
        {-3.0, -1.0, 0.0, 1.0, 3.0},
        -5.0, 5.0,
        1e-10,  // pdf_tolerance (auto-vectorisable fabs; no SIMD approximation error)
        1e-10   // cdf_tolerance
    };
    cfg.invalid_scenarios = {
        {"b = 0",  [] { return LaplaceDistribution::create(0.0, 0.0).isError(); }},
        {"b < 0",  [] { return LaplaceDistribution::create(0.0, -1.0).isError(); }},
        {"mu = inf", [] { return LaplaceDistribution::create(
                               std::numeric_limits<double>::infinity(), 1.0).isError(); }},
        {"b = NaN",  [] { return LaplaceDistribution::create(
                               0.0, std::numeric_limits<double>::quiet_NaN()).isError(); }},
    };

    try {
        // Test 1: Constructors
        BasicTestFormatter::printTestStart(1, "Constructors and Destructor");
        cout << "Laplace(mu, b): double-exponential, symmetric about mu.\n";

        auto def = LaplaceDistribution::create().unwrap();
        BasicTestFormatter::printProperty("Default mu (expect 0)", def.getMu());
        BasicTestFormatter::printProperty("Default b  (expect 1)", def.getB());

        auto std_lap = LaplaceDistribution::create(0.0, 1.0).unwrap();
        auto lap_5_2 = LaplaceDistribution::create(5.0, 2.0).unwrap();
        BasicTestFormatter::printProperty("Lap(5,2) mean (expect 5)", lap_5_2.getMean());
        BasicTestFormatter::printProperty("Lap(5,2) isStandard (expect 0)", lap_5_2.isStandard());
        BasicTestFormatter::printProperty("Lap(0,1) isStandard (expect 1)", std_lap.isStandard());

        auto copy_l = std_lap;
        auto move_l = std::move(copy_l);
        BasicTestFormatter::printProperty("Copy/move mu", move_l.getMu());
        BasicTestFormatter::printTestSuccess("Constructors passed");
        BasicTestFormatter::printNewline();

        // Test 2: Parameter getters and setters
        BasicTestFormatter::printTestStart(2, "Parameter Getters and Setters");
        auto l = LaplaceDistribution::create(1.0, 2.0).unwrap();
        BasicTestFormatter::printProperty("getMu()", l.getMu());
        BasicTestFormatter::printProperty("getB()",  l.getB());
        BasicTestFormatter::printProperty("getMuAtomic()", l.getMuAtomic());
        BasicTestFormatter::printProperty("getBAtomic()",  l.getBAtomic());

        l.setMu(-1.0);
        BasicTestFormatter::printProperty("After setMu(-1)", l.getMu());
        l.setB(0.5);
        BasicTestFormatter::printProperty("After setB(0.5)", l.getB());
        l.setParameters(3.0, 1.0);
        BasicTestFormatter::printProperty("After setParameters(3,1) mu", l.getMu());

        auto r1 = l.trySetMu(0.0);
        cout << "trySetMu(0) ok: " << (r1.isOk() ? "YES" : "NO") << "\n";
        auto r2 = l.trySetB(-1.0);
        cout << "trySetB(-1) isError: " << (r2.isError() ? "YES" : "NO") << "\n";

        BasicTestFormatter::printTestSuccess("Getters/setters passed");
        BasicTestFormatter::printNewline();

        // Test 3: Core probability methods
        BasicTestFormatter::printTestStart(3, "Core Probability Methods");
        cout << "Standard Laplace(mu=0, b=1):\n";
        cout << "  PDF(0) = 1/(2*1) = 0.5; LogPDF(0) = -log(2) ≈ -0.6931\n";
        cout << "  CDF(0) = 0.5 (exactly, by symmetry)\n";
        auto sl = LaplaceDistribution::create(0.0, 1.0).unwrap();

        double pdf0 = sl.getProbability(0.0);
        cout << "PDF(0) = " << pdf0 << " [expect 0.5]\n";
        if (std::abs(pdf0 - 0.5) > 1e-12) throw runtime_error("PDF(0) failed");

        double lp0 = sl.getLogProbability(0.0);
        cout << "LogPDF(0) = " << lp0 << " [expect " << -std::log(2.0) << "]\n";
        if (std::abs(lp0 - (-std::log(2.0))) > 1e-12) throw runtime_error("LogPDF(0) failed");

        double cdf0 = sl.getCumulativeProbability(0.0);
        cout << "CDF(0) = " << cdf0 << " [expect 0.5]\n";
        if (std::abs(cdf0 - 0.5) > 1e-12) throw runtime_error("CDF(0) failed");

        // CDF at x=1: 1 - 0.5*exp(-1) ≈ 0.8161
        double cdf1 = sl.getCumulativeProbability(1.0);
        double expect_cdf1 = 1.0 - 0.5 * std::exp(-1.0);
        cout << "CDF(1) = " << cdf1 << " [expect " << expect_cdf1 << "]\n";
        if (std::abs(cdf1 - expect_cdf1) > 1e-12) throw runtime_error("CDF(1) failed");

        // Quantile round-trip
        double q025 = sl.getQuantile(0.25);
        cout << "Q(0.25) = " << q025 << " [expect " << std::log(0.5) << "]\n";  // log(2*0.25)=log(0.5)
        cout << "CDF(Q(0.25)) = " << sl.getCumulativeProbability(q025) << " [expect 0.25]\n";

        // Moments
        cout << "Mean = " << sl.getMean() << " [expect 0.0]\n";
        cout << "Variance = " << sl.getVariance() << " [expect 2.0]\n";
        cout << "Skewness = " << sl.getSkewness() << " [expect 0.0]\n";
        cout << "Kurtosis = " << sl.getKurtosis() << " [expect 3.0]\n";
        cout << "Entropy  = " << sl.getEntropy() << " [expect 1+log(2) ≈ 1.693]\n";
        cout << "Median   = " << sl.getMedian() << " [expect 0.0]\n";
        cout << "Mode     = " << sl.getMode() << " [expect 0.0]\n";

        if (std::abs(sl.getMean() - 0.0) > 1e-12) throw runtime_error("Mean failed");
        if (std::abs(sl.getVariance() - 2.0) > 1e-12) throw runtime_error("Variance failed");
        if (std::abs(sl.getSkewness()) > 1e-12) throw runtime_error("Skewness failed");
        if (std::abs(sl.getKurtosis() - 3.0) > 1e-12) throw runtime_error("Kurtosis failed");

        // Symmetry: PDF(mu+d) == PDF(mu-d) for any d
        auto lap = LaplaceDistribution::create(2.0, 1.5).unwrap();
        for (double d : {0.5, 1.0, 2.5}) {
            double lo = lap.getProbability(2.0 - d);
            double hi = lap.getProbability(2.0 + d);
            if (std::abs(lo - hi) > 1e-12)
                throw runtime_error("Symmetry violated");
        }
        cout << "Symmetry check: PASS\n";
        cout << "isDiscrete: " << (sl.isDiscrete() ? "YES" : "NO") << "\n";

        BasicTestFormatter::printTestSuccess("Core probability methods passed");
        BasicTestFormatter::printNewline();

        // Test 4: Random Sampling
        BasicTestFormatter::printTestStart(4, "Random Sampling");
        mt19937 rng(42);
        auto lap4 = LaplaceDistribution::create(3.0, 2.0).unwrap();  // mean=3

        double s = lap4.sample(rng);
        cout << "Single sample: " << s << "\n";
        if (!std::isfinite(s)) throw runtime_error("Sample not finite");

        auto samples = lap4.sample(rng, 500);
        double smean = 0.0;
        for (double sv : samples) smean += sv;
        smean /= 500.0;
        cout << "Sample mean (n=500, expect ~3.0): " << smean << "\n";

        BasicTestFormatter::printTestSuccess("Sampling tests passed");
        BasicTestFormatter::printNewline();

        // Test 5: Distribution Management
        BasicTestFormatter::printTestStart(5, "Distribution Management");
        cout << "MLE: mu_hat = median, b_hat = mean|xi - mu_hat|  (O(n log n))\n";

        auto source = LaplaceDistribution::create(2.0, 0.5).unwrap();
        auto fit_data = source.sample(rng, 500);
        auto l5 = LaplaceDistribution::create().unwrap();
        l5.fit(fit_data);
        cout << "Fitted mu (from Lap(2, 0.5), expect ~2): " << l5.getMu() << "\n";
        cout << "Fitted b  (from Lap(2, 0.5), expect ~0.5): " << l5.getB() << "\n";

        l5.reset();
        if (std::abs(l5.getMu()) > 1e-10 || std::abs(l5.getB() - 1.0) > 1e-10)
            throw runtime_error("Reset failed");
        cout << "After reset: mu=0, b=1 (PASS)\n";
        cout << "toString: " << l5.toString() << "\n";

        BasicTestFormatter::printTestSuccess("Distribution management passed");
        BasicTestFormatter::printNewline();

        // Tests 6 and 8
        auto l6 = LaplaceDistribution::create(0.0, 1.0).unwrap();
        stats::tests::runBatchTests(cfg, l6);

        // Test 7: Comparison and Stream Operators
        BasicTestFormatter::printTestStart(7, "Comparison and Stream Operators");
        auto d1 = LaplaceDistribution::create(0.0, 1.0).unwrap();
        auto d2 = LaplaceDistribution::create(0.0, 1.0).unwrap();
        auto d3 = LaplaceDistribution::create(1.0, 2.0).unwrap();
        cout << "d1 == d2: " << (d1 == d2 ? "true" : "false") << "\n";
        cout << "d1 != d3: " << (d1 != d3 ? "true" : "false") << "\n";

        ostringstream oss;
        oss << d1;
        cout << "Stream: " << oss.str() << "\n";
        istringstream iss(oss.str());
        auto parsed = LaplaceDistribution::create().unwrap();
        iss >> parsed;
        if (std::abs(parsed.getMu()) > 1e-10 || std::abs(parsed.getB() - 1.0) > 1e-10)
            throw runtime_error("Stream round-trip failed");
        cout << "Stream round-trip: mu=" << parsed.getMu() << " b=" << parsed.getB() << "\n";

        BasicTestFormatter::printTestSuccess("Comparison and stream passed");
        BasicTestFormatter::printNewline();

        stats::tests::runErrorTests(cfg);

        BasicTestFormatter::printCompletionMessage("Laplace");
        BasicTestFormatter::printSummaryHeader();
        BasicTestFormatter::printSummaryItem("Standalone implementation: fabs + vector_exp pipeline");
        BasicTestFormatter::printSummaryItem("PDF = (1/2b)*exp(-|x-mu|/b); symmetric about mu");
        BasicTestFormatter::printSummaryItem("MLE: mu_hat=median, b_hat=MAD (closed form, O(n log n))");
        BasicTestFormatter::printSummaryItem("Moments: mean=mode=median=mu, variance=2b^2, skewness=0, kurtosis=3");
        BasicTestFormatter::printSummaryItem("Quantile: closed form, no iteration");

        return 0;
    } catch (const exception& e) {
        cerr << "FATAL: " << e.what() << endl;
        return 1;
    }
}

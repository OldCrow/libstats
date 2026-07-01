// Basic test for CauchyDistribution (delegation wrapper over StudentT(ν=1)).
// PDF: 1/(πγ(1+((x-x0)/γ)²)); MLE: median/IQR seed + Fisher scoring; support: all reals.
// Note: getMean/getVariance/getSkewness/getKurtosis all return NaN (moments undefined).
#include "include/basic_test_runner.h"
#include "include/tests.h"
#include "libstats/distributions/cauchy.h"

#include <cmath>
#include <iostream>
#include <random>

using namespace std;
using namespace stats;
using namespace stats::tests::fixtures;

int main() {
    BasicTestFormatter::printTestHeader("Cauchy");

    stats::tests::BasicDistConfig cfg{
        "Cauchy", {-5.0, -1.0, 0.0, 1.0, 5.0}, -10.0, 10.0,
        1e-10,  // pdf_tolerance (via StudentT delegation — no extra approx error)
        1e-10   // cdf_tolerance
    };
    cfg.invalid_scenarios = {
        {"gamma = 0", [] { return CauchyDistribution::create(0.0, 0.0).isError(); }},
        {"gamma < 0", [] { return CauchyDistribution::create(0.0, -1.0).isError(); }},
        {"x0 = inf",
         [] {
             return CauchyDistribution::create(std::numeric_limits<double>::infinity(), 1.0)
                 .isError();
         }},
        {"gamma = NaN",
         [] {
             return CauchyDistribution::create(0.0, std::numeric_limits<double>::quiet_NaN())
                 .isError();
         }},
    };

    try {
        // Test 1: Constructors
        BasicTestFormatter::printTestStart(1, "Constructors and Destructor");
        cout << "Cauchy(x0, gamma): delegates to StudentT(nu=1) with transform z=(x-x0)/gamma.\n";

        auto def = CauchyDistribution::create().unwrap();
        BasicTestFormatter::printProperty("Default x0 (expect 0)", def.getX0());
        BasicTestFormatter::printProperty("Default gamma (expect 1)", def.getGamma());

        auto std_c = CauchyDistribution::create(0.0, 1.0).unwrap();
        auto c5_2 = CauchyDistribution::create(5.0, 2.0).unwrap();
        BasicTestFormatter::printProperty("Cauchy(5,2) x0 (expect 5)", c5_2.getX0());
        BasicTestFormatter::printProperty("Cauchy(5,2) isStandard (expect 0)", c5_2.isStandard());
        BasicTestFormatter::printProperty("Cauchy(0,1) isStandard (expect 1)", std_c.isStandard());
        BasicTestFormatter::printProperty("isDiscrete (expect 0)", std_c.isDiscrete());

        auto copy_c = std_c;
        auto move_c = std::move(copy_c);
        BasicTestFormatter::printProperty("Copy/move x0", move_c.getX0());
        BasicTestFormatter::printTestSuccess("Constructors passed");
        BasicTestFormatter::printNewline();

        // Test 2: Parameter getters and setters
        BasicTestFormatter::printTestStart(2, "Parameter Getters and Setters");
        auto c = CauchyDistribution::create(1.0, 2.0).unwrap();
        BasicTestFormatter::printProperty("getX0()", c.getX0());
        BasicTestFormatter::printProperty("getGamma()", c.getGamma());
        BasicTestFormatter::printProperty("getX0Atomic()", c.getX0Atomic());
        BasicTestFormatter::printProperty("getGammaAtomic()", c.getGammaAtomic());

        c.setX0(-1.0);
        BasicTestFormatter::printProperty("After setX0(-1)", c.getX0());
        c.setGamma(0.5);
        BasicTestFormatter::printProperty("After setGamma(0.5)", c.getGamma());
        c.setParameters(3.0, 1.0);
        BasicTestFormatter::printProperty("After setParameters(3,1) x0", c.getX0());

        auto r1 = c.trySetX0(0.0);
        cout << "trySetX0(0) ok: " << (r1.isOk() ? "YES" : "NO") << "\n";
        auto r2 = c.trySetGamma(-1.0);
        cout << "trySetGamma(-1) isError: " << (r2.isError() ? "YES" : "NO") << "\n";

        BasicTestFormatter::printTestSuccess("Getters/setters passed");
        BasicTestFormatter::printNewline();

        // Test 3: Core probability methods
        BasicTestFormatter::printTestStart(3, "Core Probability Methods");
        cout << "Standard Cauchy(x0=0, gamma=1):\n";
        cout << "  PDF(0) = 1/pi ≈ 0.3183; CDF(0) = 0.5 (exactly)\n";
        auto sc = CauchyDistribution::create(0.0, 1.0).unwrap();

        const double expected_pdf0 = 1.0 / detail::PI;
        double pdf0 = sc.getProbability(0.0);
        cout << "PDF(0) = " << pdf0 << " [expect " << expected_pdf0 << "]\n";
        if (std::abs(pdf0 - expected_pdf0) > 1e-10)
            throw runtime_error("PDF(0) failed");

        const double expected_lpdf0 = -std::log(detail::PI);
        double lpdf0 = sc.getLogProbability(0.0);
        cout << "LogPDF(0) = " << lpdf0 << " [expect " << expected_lpdf0 << "]\n";
        if (std::abs(lpdf0 - expected_lpdf0) > 1e-10)
            throw runtime_error("LogPDF(0) failed");

        double cdf0 = sc.getCumulativeProbability(0.0);
        cout << "CDF(0) = " << cdf0 << " [expect 0.5]\n";
        if (std::abs(cdf0 - 0.5) > 1e-10)
            throw runtime_error("CDF(0) failed");

        // Quantile round-trip
        for (double p : {0.1, 0.25, 0.5, 0.75, 0.9}) {
            double q = sc.getQuantile(p);
            double cdfq = sc.getCumulativeProbability(q);
            if (std::abs(cdfq - p) > 1e-8)
                throw runtime_error("Quantile round-trip failed at p=" + to_string(p));
        }
        cout << "Quantile round-trip: PASS\n";

        // Moments: all NaN except median/mode
        cout << "getMean() = " << sc.getMean() << " [expect NaN]\n";
        cout << "getVariance() = " << sc.getVariance() << " [expect NaN]\n";
        cout << "getSkewness() = " << sc.getSkewness() << " [expect NaN]\n";
        cout << "getKurtosis() = " << sc.getKurtosis() << " [expect NaN]\n";
        cout << "getMedian() = " << sc.getMedian() << " [expect 0.0]\n";
        cout << "getMode() = " << sc.getMode() << " [expect 0.0]\n";
        cout << "getEntropy() = " << sc.getEntropy() << " [expect log(4*pi) ≈ "
             << std::log(detail::FOUR_PI) << "]\n";

        if (!std::isnan(sc.getMean()))
            throw runtime_error("getMean should be NaN");
        if (!std::isnan(sc.getVariance()))
            throw runtime_error("getVariance should be NaN");
        if (!std::isnan(sc.getSkewness()))
            throw runtime_error("getSkewness should be NaN");
        if (!std::isnan(sc.getKurtosis()))
            throw runtime_error("getKurtosis should be NaN");
        if (std::abs(sc.getMedian()) > 1e-10)
            throw runtime_error("Median failed");
        if (std::abs(sc.getMode()) > 1e-10)
            throw runtime_error("Mode failed");
        if (std::abs(sc.getEntropy() - std::log(detail::FOUR_PI)) > 1e-10)
            throw runtime_error("Entropy failed");

        // Symmetry: PDF(x0+d) == PDF(x0-d) for any d
        auto cshift = CauchyDistribution::create(2.0, 1.5).unwrap();
        for (double d : {0.5, 1.0, 2.5, 5.0}) {
            double lo = cshift.getProbability(2.0 - d);
            double hi = cshift.getProbability(2.0 + d);
            if (std::abs(lo - hi) > 1e-12)
                throw runtime_error("PDF symmetry violated at d=" + to_string(d));
        }
        cout << "PDF symmetry check: PASS\n";

        // NaN propagation
        const double nan_val = std::numeric_limits<double>::quiet_NaN();
        if (!std::isnan(sc.getProbability(nan_val)))
            throw runtime_error("PDF(NaN) should be NaN");
        if (!std::isnan(sc.getLogProbability(nan_val)))
            throw runtime_error("LogPDF(NaN) should be NaN");
        if (!std::isnan(sc.getCumulativeProbability(nan_val)))
            throw runtime_error("CDF(NaN) should be NaN");
        cout << "NaN propagation: PASS\n";

        BasicTestFormatter::printTestSuccess("Core probability methods passed");
        BasicTestFormatter::printNewline();

        // Test 4: Random Sampling
        BasicTestFormatter::printTestStart(4, "Random Sampling");
        mt19937 rng(42);
        auto c4 = CauchyDistribution::create(1.0, 2.0).unwrap();

        double s = c4.sample(rng);
        cout << "Single sample: " << s << "\n";
        if (!std::isfinite(s))
            throw runtime_error("Sample not finite");

        auto samples = c4.sample(rng, 500);
        if (samples.size() != 500)
            throw runtime_error("Sample count wrong");
        for (double sv : samples)
            if (!std::isfinite(sv))
                throw runtime_error("Sample not finite");
        cout << "500 samples: all finite (PASS)\n";

        BasicTestFormatter::printTestSuccess("Sampling tests passed");
        BasicTestFormatter::printNewline();

        // Test 5: Distribution Management
        BasicTestFormatter::printTestStart(5, "Distribution Management");
        cout << "MLE: median seed, IQR/2 seed, Fisher-scoring iterations\n";

        auto source = CauchyDistribution::create(3.0, 1.5).unwrap();
        auto fit_data = source.sample(rng, 1000);
        auto c5 = CauchyDistribution::create().unwrap();
        c5.fit(fit_data);
        cout << "Fitted x0 (from Cauchy(3,1.5), expect ~3): " << c5.getX0() << "\n";
        cout << "Fitted gamma (from Cauchy(3,1.5), expect ~1.5): " << c5.getGamma() << "\n";

        c5.reset();
        if (std::abs(c5.getX0()) > 1e-10 || std::abs(c5.getGamma() - 1.0) > 1e-10)
            throw runtime_error("Reset failed");
        cout << "After reset: x0=0, gamma=1 (PASS)\n";
        cout << "toString: " << c5.toString() << "\n";

        BasicTestFormatter::printTestSuccess("Distribution management passed");
        BasicTestFormatter::printNewline();

        // Tests 6 and 8
        auto c6 = CauchyDistribution::create(0.0, 1.0).unwrap();
        stats::tests::runBatchTests(cfg, c6);

        // Test 7: Comparison and Stream Operators
        BasicTestFormatter::printTestStart(7, "Comparison and Stream Operators");
        auto d1 = CauchyDistribution::create(0.0, 1.0).unwrap();
        auto d2 = CauchyDistribution::create(0.0, 1.0).unwrap();
        auto d3 = CauchyDistribution::create(1.0, 2.0).unwrap();
        cout << "d1 == d2: " << (d1 == d2 ? "true" : "false") << "\n";
        cout << "d1 != d3: " << (d1 != d3 ? "true" : "false") << "\n";

        ostringstream oss;
        oss << d1;
        cout << "Stream: " << oss.str() << "\n";
        istringstream iss(oss.str());
        auto parsed = CauchyDistribution::create().unwrap();
        iss >> parsed;
        if (std::abs(parsed.getX0()) > 1e-8 || std::abs(parsed.getGamma() - 1.0) > 1e-8)
            throw runtime_error("Stream round-trip failed");
        cout << "Stream round-trip: x0=" << parsed.getX0() << " gamma=" << parsed.getGamma()
             << "\n";

        BasicTestFormatter::printTestSuccess("Comparison and stream passed");
        BasicTestFormatter::printNewline();

        stats::tests::runErrorTests(cfg);

        BasicTestFormatter::printCompletionMessage("Cauchy");
        BasicTestFormatter::printSummaryHeader();
        BasicTestFormatter::printSummaryItem("Delegation wrapper: delegates to StudentT(nu=1)");
        BasicTestFormatter::printSummaryItem(
            "PDF = 1/(pi*gamma*(1+((x-x0)/gamma)^2)); symmetric about x0");
        BasicTestFormatter::printSummaryItem(
            "MLE: median seed, IQR/2 seed, 20 Fisher-scoring iterations");
        BasicTestFormatter::printSummaryItem(
            "Moments: mean/variance/skewness/kurtosis all NaN; median=mode=x0");
        BasicTestFormatter::printSummaryItem("Quantile: closed form x0+gamma*tan(pi*(p-0.5))");

        return 0;
    } catch (const exception& e) {
        cerr << "FATAL: " << e.what() << endl;
        return 1;
    }
}

// Focused unit test for Erlang distribution
#include "include/basic_test_runner.h"
#include "include/tests.h"
#include "libstats/distributions/erlang.h"

#include <cmath>
#include <iomanip>
#include <iostream>
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
    BasicTestFormatter::printTestHeader("Erlang");

    try {
        // =====================================================================
        // Test 1: Constructors and Destructor
        // =====================================================================
        BasicTestFormatter::printTestStart(1, "Constructors and Destructor");
        cout << "Erlang is a delegation wrapper: Erlang(k, lambda) = Gamma(k, lambda)." << endl;
        cout << "libstats' Gamma is (alpha shape, beta RATE), so Erlang delegates directly."
             << endl;

        auto default_erl = stats::ErlangDistribution::create().unwrap();
        BasicTestFormatter::printPropertyInt("Default k (shape)", default_erl.getK());
        BasicTestFormatter::printProperty("Default lambda (rate)", default_erl.getLambda());
        BasicTestFormatter::printProperty("Default mean (should be 1)", default_erl.getMean());

        auto erl_k2 = stats::ErlangDistribution::create(2, 3.0).unwrap();
        BasicTestFormatter::printPropertyInt("k=2, lambda=3 distribution created",
                                             erl_k2.getK());

        auto copy_erl = erl_k2;
        BasicTestFormatter::printPropertyInt("Copy k", copy_erl.getK());

        auto temp = stats::ErlangDistribution::create(5, 2.0).unwrap();
        auto move_erl = std::move(temp);
        BasicTestFormatter::printPropertyInt("Move k", move_erl.getK());

        auto result = ErlangDistribution::create(3, 1.0);
        if (result.isOk()) {
            BasicTestFormatter::printPropertyInt("Factory k=3", (*result).getK());
        }

        BasicTestFormatter::printTestSuccess("All constructor tests passed");
        BasicTestFormatter::printNewline();

        // =====================================================================
        // Test 2: Parameter Getters and Setters
        // =====================================================================
        BasicTestFormatter::printTestStart(2, "Parameter Getters and Setters");

        auto erl = stats::ErlangDistribution::create(4, 2.0).unwrap();

        BasicTestFormatter::printPropertyInt("Initial k", erl.getK());
        BasicTestFormatter::printProperty("Initial lambda", erl.getLambda());
        BasicTestFormatter::printProperty("Mean (should be 2)", erl.getMean());
        BasicTestFormatter::printProperty("Variance (should be 1)", erl.getVariance());
        BasicTestFormatter::printPropertyInt("Num parameters (should be 2)",
                                             erl.getNumParameters());
        cout << "Distribution name: " << erl.getDistributionName() << endl;
        cout << "Is discrete: " << (erl.isDiscrete() ? "YES" : "NO") << endl;
        BasicTestFormatter::printProperty("Support lower", erl.getSupportLowerBound());
        BasicTestFormatter::printProperty("Support upper", erl.getSupportUpperBound());

        erl.setK(6);
        BasicTestFormatter::printPropertyInt("After setK(6): k", erl.getK());

        erl.setLambda(3.0);
        BasicTestFormatter::printProperty("After setLambda(3): lambda", erl.getLambda());
        BasicTestFormatter::printProperty("Mean (should be 2)", erl.getMean());

        auto set_result = erl.trySetParameters(5, 1.0);
        if (set_result.isOk()) {
            BasicTestFormatter::printPropertyInt("trySetParameters(5,1.0): k", erl.getK());
        }

        auto bad_result = erl.trySetK(0);
        cout << "trySetK(0) error (expected): " << bad_result.message() << endl;

        BasicTestFormatter::printTestSuccess("All setter/getter tests passed");
        BasicTestFormatter::printNewline();

        // =====================================================================
        // Test 3: Core Probability Methods (known numerical values)
        // =====================================================================
        BasicTestFormatter::printTestStart(3, "Core Probability Methods");
        cout << "Erlang(k=1, lambda=2) = Exponential(2): PDF(x) = 2*exp(-2x)" << endl;

        auto erl_exp = stats::ErlangDistribution::create(1, 2.0).unwrap();

        const double pdf_at_1 = erl_exp.getProbability(1.0);
        const double expected_pdf_at_1 = 2.0 * std::exp(-2.0);
        BasicTestFormatter::printProperty("PDF(1)   computed", pdf_at_1);
        BasicTestFormatter::printProperty("PDF(1)   expected", expected_pdf_at_1);
        const bool pdf_ok = std::abs(pdf_at_1 - expected_pdf_at_1) < 1e-10;
        cout << "PDF(1) match: " << (pdf_ok ? "PASS" : "FAIL") << endl;

        const double cdf_at_1 = erl_exp.getCumulativeProbability(1.0);
        const double expected_cdf_at_1 = 1.0 - std::exp(-2.0);
        BasicTestFormatter::printProperty("CDF(1)   computed", cdf_at_1);
        BasicTestFormatter::printProperty("CDF(1)   expected (1-e^-2)", expected_cdf_at_1);
        const bool cdf_ok = std::abs(cdf_at_1 - expected_cdf_at_1) < 1e-8;
        cout << "CDF(1) match: " << (cdf_ok ? "PASS" : "FAIL") << endl;

        // Out-of-support
        BasicTestFormatter::printProperty("PDF(-1) should be 0", erl_exp.getProbability(-1.0));
        BasicTestFormatter::printProperty("CDF(0)  should be 0",
                                          erl_exp.getCumulativeProbability(0.0));

        // ±inf contract (#103): PDF(+inf) = 0
        BasicTestFormatter::printProperty("PDF(+inf) should be 0",
                                          erl_exp.getProbability(
                                              std::numeric_limits<double>::infinity()));

        const double q50 = erl_exp.getQuantile(0.5);
        BasicTestFormatter::printProperty("Quantile(0.50)", q50);

        BasicTestFormatter::printProperty("Mode (k=1: (1-1)/2=0)", erl_exp.getMode());
        BasicTestFormatter::printProperty("Median", erl_exp.getMedian());
        BasicTestFormatter::printProperty("Entropy", erl_exp.getEntropy());

        if (!pdf_ok || !cdf_ok) {
            throw std::runtime_error("Numerical accuracy check failed");
        }

        BasicTestFormatter::printTestSuccess("All probability method tests passed");
        BasicTestFormatter::printNewline();

        // =====================================================================
        // Test 4: Random Sampling
        // =====================================================================
        BasicTestFormatter::printTestStart(4, "Random Sampling");
        cout << "Samples delegated to Gamma(k, lambda). Sample mean should ~ k/lambda." << endl;

        mt19937 rng(42);
        auto erl_4 = stats::ErlangDistribution::create(4, 2.0).unwrap();  // mean=2, var=1

        const double single = erl_4.sample(rng);
        BasicTestFormatter::printProperty("Single sample (k=4, lambda=2)", single);

        const auto samples = erl_4.sample(rng, 200);
        const double smean = TestDataGenerators::computeSampleMean(samples);
        BasicTestFormatter::printProperty("Sample mean (n=200, expect ~2)", smean);

        BasicTestFormatter::printTestSuccess("Sampling tests passed");
        BasicTestFormatter::printNewline();

        // =====================================================================
        // Test 5: Distribution Management (fit, reset, toString)
        // =====================================================================
        BasicTestFormatter::printTestStart(5, "Distribution Management");
        cout << "MoM for Erlang: k_hat = round(mean^2/var), lambda_hat = k_hat/mean." << endl;

        auto erl_fit = stats::ErlangDistribution::create(1, 1.0).unwrap();

        const auto fit_data = erl_4.sample(rng, 500);
        erl_fit.fit(fit_data);
        BasicTestFormatter::printPropertyInt("Fitted k (from Erlang(4,2) data, expect ~4)",
                                             erl_fit.getK());
        BasicTestFormatter::printProperty("Fitted lambda (expect ~2)", erl_fit.getLambda());

        erl_fit.reset();
        BasicTestFormatter::printPropertyInt("After reset: k (expect 1)", erl_fit.getK());
        BasicTestFormatter::printProperty("After reset: lambda (expect 1)", erl_fit.getLambda());

        cout << "toString: " << erl_fit.toString() << endl;

        BasicTestFormatter::printTestSuccess("Distribution management tests passed");
        BasicTestFormatter::printNewline();

        // =====================================================================
        // Test 6: Auto-dispatch Batch Operations
        // =====================================================================
        stats::tests::BasicDistConfig cfg{"Erlang", {0.5, 1.0, 2.0, 4.0, 8.0}, 0.1, 10.0,
                                          1e-12,  // pdf_tolerance
                                          1e-12};
        auto erl_batch = stats::ErlangDistribution::create(3, 1.0).unwrap();
        stats::tests::runBatchTests(cfg, erl_batch);

        BasicTestFormatter::printTestStart(7, "Comparison and Stream Operators");

        auto a = stats::ErlangDistribution::create(3, 1.0).unwrap();
        auto b = stats::ErlangDistribution::create(3, 1.0).unwrap();
        auto c = stats::ErlangDistribution::create(5, 1.0).unwrap();

        cout << "a==b (k=3 vs k=3): " << (a == b ? "true" : "false") << endl;
        cout << "a!=c (k=3 vs k=5): " << (a != c ? "true" : "false") << endl;

        ostringstream oss;
        oss << a;
        cout << "Stream output: " << oss.str() << endl;

        istringstream iss("ErlangDistribution(k=7,lambda=2.5)");
        ErlangDistribution parsed = stats::ErlangDistribution::create().unwrap();
        iss >> parsed;
        BasicTestFormatter::printPropertyInt("Parsed from stream: k (expect 7)", parsed.getK());
        BasicTestFormatter::printProperty("Parsed from stream: lambda (expect 2.5)",
                                          parsed.getLambda());
        if (parsed.getK() != 7 || std::abs(parsed.getLambda() - 2.5) > 1e-10)
            throw std::runtime_error("Stream round-trip failed");

        BasicTestFormatter::printTestSuccess("Comparison and stream tests passed");
        BasicTestFormatter::printNewline();

        // =====================================================================
        // Test 8: Error Handling
        // =====================================================================
        BasicTestFormatter::printTestStart(8, "Error Handling");

        auto err_zero_k = ErlangDistribution::create(0, 1.0);
        cout << "create(0, 1.0)  isError(): " << (err_zero_k.isError() ? "YES" : "NO") << endl;

        auto err_neg_k = ErlangDistribution::create(-1, 1.0);
        cout << "create(-1, 1.0) isError(): " << (err_neg_k.isError() ? "YES" : "NO") << endl;

        auto err_zero_lambda = ErlangDistribution::create(1, 0.0);
        cout << "create(1, 0.0)  isError(): " << (err_zero_lambda.isError() ? "YES" : "NO")
             << endl;

        auto err_nan = ErlangDistribution::create(1, std::numeric_limits<double>::quiet_NaN());
        cout << "create(1, NaN)  isError(): " << (err_nan.isError() ? "YES" : "NO") << endl;

        auto err_inf = ErlangDistribution::create(1, std::numeric_limits<double>::infinity());
        cout << "create(1, inf)  isError(): " << (err_inf.isError() ? "YES" : "NO") << endl;

        auto erl_err = ErlangDistribution::create(3, 1.0).unwrap();
        auto vr = erl_err.trySetK(-5);
        cout << "trySetK(-5)  isError(): " << (vr.isError() ? "YES" : "NO") << endl;
        BasicTestFormatter::printPropertyInt("k unchanged after failed trySetK (expect 3)",
                                             erl_err.getK());

        if (!err_zero_k.isError() || !err_neg_k.isError() || !err_zero_lambda.isError() ||
            !err_nan.isError() || !err_inf.isError() || !vr.isError() || erl_err.getK() != 3) {
            throw std::runtime_error("Error handling test failed");
        }

        BasicTestFormatter::printTestSuccess("Error handling tests passed");
        BasicTestFormatter::printNewline();

        BasicTestFormatter::printTestHeader("Erlang - ALL TESTS PASSED");

    } catch (const std::exception& e) {
        cerr << "FATAL: " << e.what() << endl;
        return 1;
    }

    return 0;
}

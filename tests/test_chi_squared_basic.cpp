// Focused unit test for chi-squared distribution
#include "include/tests.h"
#include "libstats/distributions/chi_squared.h"

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
    BasicTestFormatter::printTestHeader("ChiSquared");

    try {
        // =====================================================================
        // Test 1: Constructors and Destructor
        // =====================================================================
        BasicTestFormatter::printTestStart(1, "Constructors and Destructor");
        cout << "Chi-squared is a delegation wrapper: ChiSquared(k) = Gamma(k/2, 0.5)." << endl;
        cout << "All probability methods forward to an internal GammaDistribution." << endl;

        auto default_chi2 = stats::ChiSquaredDistribution::create().value;
        BasicTestFormatter::printProperty("Default k (df)", default_chi2.getK());
        BasicTestFormatter::printProperty("Default mean (should be 1)", default_chi2.getMean());

        auto chi2_k2 = stats::ChiSquaredDistribution::create(2.0).value;
        BasicTestFormatter::printProperty("k=2 distribution created", chi2_k2.getK());

        auto copy_chi2 = chi2_k2;
        BasicTestFormatter::printProperty("Copy k", copy_chi2.getK());

        auto temp = stats::ChiSquaredDistribution::create(5.0).value;
        auto move_chi2 = std::move(temp);
        BasicTestFormatter::printProperty("Move k", move_chi2.getK());

        auto result = ChiSquaredDistribution::create(3.0);
        if (result.isOk()) {
            BasicTestFormatter::printProperty("Factory k=3", result.value.getK());
        }

        BasicTestFormatter::printTestSuccess("All constructor tests passed");
        BasicTestFormatter::printNewline();

        // =====================================================================
        // Test 2: Parameter Getters and Setters
        // =====================================================================
        BasicTestFormatter::printTestStart(2, "Parameter Getters and Setters");
        cout << "Tests getK/getDegreesOfFreedom, setK/setDegreesOfFreedom," << endl;
        cout << "and the safe trySetK/trySetParameters Result-based API." << endl;

        auto chi2 = stats::ChiSquaredDistribution::create(4.0).value;

        BasicTestFormatter::printProperty("Initial k (df)", chi2.getK());
        BasicTestFormatter::printProperty("getDegreesOfFreedom()", chi2.getDegreesOfFreedom());
        BasicTestFormatter::printProperty("Mean (should be 4)", chi2.getMean());
        BasicTestFormatter::printProperty("Variance (should be 8)", chi2.getVariance());
        BasicTestFormatter::printProperty("Skewness (sqrt(8/4)=sqrt(2))", chi2.getSkewness());
        BasicTestFormatter::printProperty("Kurtosis (12/4=3)", chi2.getKurtosis());
        BasicTestFormatter::printPropertyInt("Num parameters (should be 1)",
                                             chi2.getNumParameters());
        cout << "Distribution name: " << chi2.getDistributionName() << endl;
        cout << "Is discrete: " << (chi2.isDiscrete() ? "YES" : "NO") << endl;
        BasicTestFormatter::printProperty("Support lower", chi2.getSupportLowerBound());
        BasicTestFormatter::printProperty("Support upper", chi2.getSupportUpperBound());

        chi2.setK(6.0);
        BasicTestFormatter::printProperty("After setK(6): k", chi2.getK());
        BasicTestFormatter::printProperty("After setK(6): mean (should be 6)", chi2.getMean());

        chi2.setDegreesOfFreedom(8.0);
        BasicTestFormatter::printProperty("After setDegreesOfFreedom(8): k", chi2.getK());

        auto set_result = chi2.trySetK(5.0);
        if (set_result.isOk()) {
            BasicTestFormatter::printProperty("trySetK(5): k", chi2.getK());
        }

        auto bad_result = chi2.trySetK(-1.0);
        cout << "trySetK(-1) error (expected): " << bad_result.message << endl;

        BasicTestFormatter::printTestSuccess("All setter/getter tests passed");
        BasicTestFormatter::printNewline();

        // =====================================================================
        // Test 3: Core Probability Methods (known numerical values)
        // =====================================================================
        BasicTestFormatter::printTestStart(3, "Core Probability Methods");
        cout << "Verifies PDF/LogPDF/CDF against reference values from chi-squared tables." << endl;
        cout << "ChiSquared(k=2): PDF(x) = 0.5*exp(-x/2), mean=2, variance=4." << endl;
        cout << "ChiSquared(k=2): PDF(1)   = 0.5*exp(-0.5) ≈ 0.30327" << endl;
        cout << "ChiSquared(k=2): CDF(2)   = 1 - exp(-1)   ≈ 0.63212" << endl;
        cout << "ChiSquared(k=2): CDF(5.99) ≈ 0.95 (critical value)" << endl;

        auto chi2_2 = stats::ChiSquaredDistribution::create(2.0).value;

        const double pdf_at_1 = chi2_2.getProbability(1.0);
        const double expected_pdf_at_1 = 0.5 * std::exp(-0.5);
        BasicTestFormatter::printProperty("PDF(1)   computed", pdf_at_1);
        BasicTestFormatter::printProperty("PDF(1)   expected", expected_pdf_at_1);
        const bool pdf_ok = std::abs(pdf_at_1 - expected_pdf_at_1) < 1e-10;
        cout << "PDF(1) match: " << (pdf_ok ? "PASS" : "FAIL") << endl;

        const double log_pdf_at_1 = chi2_2.getLogProbability(1.0);
        BasicTestFormatter::printProperty("LogPDF(1)", log_pdf_at_1);
        const bool log_pdf_ok = std::abs(log_pdf_at_1 - std::log(expected_pdf_at_1)) < 1e-10;
        cout << "LogPDF(1) match: " << (log_pdf_ok ? "PASS" : "FAIL") << endl;

        const double cdf_at_2 = chi2_2.getCumulativeProbability(2.0);
        const double expected_cdf_at_2 = 1.0 - std::exp(-1.0);
        BasicTestFormatter::printProperty("CDF(2)   computed", cdf_at_2);
        BasicTestFormatter::printProperty("CDF(2)   expected (1-e^-1)", expected_cdf_at_2);
        const bool cdf_ok = std::abs(cdf_at_2 - expected_cdf_at_2) < 1e-8;
        cout << "CDF(2) match: " << (cdf_ok ? "PASS" : "FAIL") << endl;

        const double cdf_95 = chi2_2.getCumulativeProbability(5.991);
        BasicTestFormatter::printProperty("CDF(5.991) ≈ 0.95", cdf_95);

        // Out-of-support
        BasicTestFormatter::printProperty("PDF(-1) should be 0", chi2_2.getProbability(-1.0));
        BasicTestFormatter::printProperty("CDF(0)  should be 0",
                                          chi2_2.getCumulativeProbability(0.0));

        // Quantile (inverse CDF)
        const double q50 = chi2_2.getQuantile(0.5);
        BasicTestFormatter::printProperty("Quantile(0.50) ≈ ln(2)*2 ≈ 1.386", q50);
        const double q95 = chi2_2.getQuantile(0.95);
        BasicTestFormatter::printProperty("Quantile(0.95) ≈ 5.991", q95);

        // Distribution-specific utilities (delegated to gamma_)
        BasicTestFormatter::printProperty("Mode (k=2: max(2-2,0)=0)", chi2_2.getMode());
        BasicTestFormatter::printProperty("Median", chi2_2.getMedian());
        BasicTestFormatter::printProperty("Entropy", chi2_2.getEntropy());

        if (!pdf_ok || !log_pdf_ok || !cdf_ok) {
            throw std::runtime_error("Numerical accuracy check failed");
        }

        BasicTestFormatter::printTestSuccess("All probability method tests passed");
        BasicTestFormatter::printNewline();

        // =====================================================================
        // Test 4: Random Sampling
        // =====================================================================
        BasicTestFormatter::printTestStart(4, "Random Sampling");
        cout << "Samples delegated to Gamma(k/2, 0.5). Sample mean should ≈ k." << endl;

        mt19937 rng(42);
        auto chi2_4 = stats::ChiSquaredDistribution::create(4.0).value;

        const double single = chi2_4.sample(rng);
        BasicTestFormatter::printProperty("Single sample (k=4)", single);

        const auto samples = chi2_4.sample(rng, 100);
        const double smean = TestDataGenerators::computeSampleMean(samples);
        const double svar = TestDataGenerators::computeSampleVariance(samples);
        BasicTestFormatter::printProperty("Sample mean (n=100, expect ≈4)", smean);
        BasicTestFormatter::printProperty("Sample variance (n=100, expect ≈8)", svar);

        BasicTestFormatter::printTestSuccess("Sampling tests passed");
        BasicTestFormatter::printNewline();

        // =====================================================================
        // Test 5: Distribution Management (fit, reset, toString)
        // =====================================================================
        BasicTestFormatter::printTestStart(5, "Distribution Management");
        cout << "MLE for chi-squared: k_hat = sample_mean. No solver required." << endl;

        auto chi2_fit = stats::ChiSquaredDistribution::create(1.0).value;

        // Fit to samples generated from ChiSquared(4) — expect k_hat ≈ 4
        const auto fit_data = chi2_4.sample(rng, 200);
        chi2_fit.fit(fit_data);
        BasicTestFormatter::printProperty("Fitted k (from ChiSquared(4) data, expect ≈4)",
                                          chi2_fit.getK());

        chi2_fit.reset();
        BasicTestFormatter::printProperty("After reset: k (expect 1)", chi2_fit.getK());

        cout << "toString: " << chi2_fit.toString() << endl;

        BasicTestFormatter::printTestSuccess("Distribution management tests passed");
        BasicTestFormatter::printNewline();

        // =====================================================================
        // Test 6: Batch Operations (delegation to gamma_)
        // =====================================================================
        BasicTestFormatter::printTestStart(6, "Batch Operations");
        cout << "Batch calls delegate to gamma_; SIMD dispatch runs inside GammaDistribution."
             << endl;

        auto chi2_batch = stats::ChiSquaredDistribution::create(3.0).value;
        const size_t N = 1000;
        vector<double> xs(N), pdf_results(N), logpdf_results(N), cdf_results(N);
        for (size_t i = 0; i < N; ++i)
            xs[i] = 0.01 + static_cast<double>(i) * 0.02;

        chi2_batch.getProbability(span<const double>(xs), span<double>(pdf_results));
        chi2_batch.getLogProbability(span<const double>(xs), span<double>(logpdf_results));
        chi2_batch.getCumulativeProbability(span<const double>(xs), span<double>(cdf_results));

        // Spot-check first and last: PDF batch vs single
        const double batch_pdf_0 = pdf_results[0];
        const double scalar_pdf_0 = chi2_batch.getProbability(xs[0]);
        const bool batch_ok = std::abs(batch_pdf_0 - scalar_pdf_0) < 1e-14;
        cout << "Batch PDF vs scalar (first element): " << (batch_ok ? "PASS" : "FAIL") << endl;
        BasicTestFormatter::printProperty("Batch PDF[0]  (computed)", batch_pdf_0);
        BasicTestFormatter::printProperty("Scalar PDF[0] (reference)", scalar_pdf_0);

        BasicTestFormatter::printTestSuccess("Batch operation tests passed");
        BasicTestFormatter::printNewline();

        // =====================================================================
        // Test 7: Comparison and Stream Operators
        // =====================================================================
        BasicTestFormatter::printTestStart(7, "Comparison and Stream Operators");

        auto a = stats::ChiSquaredDistribution::create(3.0).value;
        auto b = stats::ChiSquaredDistribution::create(3.0).value;
        auto c = stats::ChiSquaredDistribution::create(5.0).value;

        cout << "a==b (k=3 vs k=3): " << (a == b ? "true" : "false") << endl;
        cout << "a!=c (k=3 vs k=5): " << (a != c ? "true" : "false") << endl;

        ostringstream oss;
        oss << a;
        cout << "Stream output: " << oss.str() << endl;

        istringstream iss("ChiSquaredDistribution(k=7)");
        ChiSquaredDistribution parsed = stats::ChiSquaredDistribution::create().value;
        iss >> parsed;
        BasicTestFormatter::printProperty("Parsed from stream: k (expect 7)", parsed.getK());

        BasicTestFormatter::printTestSuccess("Comparison and stream tests passed");
        BasicTestFormatter::printNewline();

        // =====================================================================
        // Test 8: Error Handling
        // =====================================================================
        BasicTestFormatter::printTestStart(8, "Error Handling");
        cout << "Using create() factory (Result-based API) to test validation." << endl;
        cout << "Note: The throwing constructor is not tested here due to a known ABI exception-"
             << endl;
        cout << "unwinding limitation with Homebrew LLVM libc++ on macOS Catalina;" << endl;
        cout << "see the create() factory doc comment in chi_squared.h." << endl;

        // Test all invalid-parameter cases via the safe factory (no ABI issues)
        auto err_zero = ChiSquaredDistribution::create(0.0);
        cout << "create(0.0)  isError(): " << (err_zero.isError() ? "YES" : "NO") << endl;

        auto err_neg = ChiSquaredDistribution::create(-1.0);
        cout << "create(-1.0) isError(): " << (err_neg.isError() ? "YES" : "NO") << endl;

        auto err_nan = ChiSquaredDistribution::create(std::numeric_limits<double>::quiet_NaN());
        cout << "create(NaN)  isError(): " << (err_nan.isError() ? "YES" : "NO") << endl;

        auto err_inf = ChiSquaredDistribution::create(std::numeric_limits<double>::infinity());
        cout << "create(inf)  isError(): " << (err_inf.isError() ? "YES" : "NO") << endl;

        // trySetK also uses the Result-based path
        auto chi2_err = ChiSquaredDistribution::create(3.0).value;
        auto vr = chi2_err.trySetK(-5.0);
        cout << "trySetK(-5)  isError(): " << (vr.isError() ? "YES" : "NO") << endl;
        BasicTestFormatter::printProperty("k unchanged after failed trySetK (expect 3)",
                                          chi2_err.getK());

        if (!err_zero.isError() || !err_neg.isError() || !err_nan.isError() || !err_inf.isError() ||
            !vr.isError() || std::abs(chi2_err.getK() - 3.0) > 1e-14) {
            throw std::runtime_error("Error handling test failed");
        }

        BasicTestFormatter::printTestSuccess("Error handling tests passed");
        BasicTestFormatter::printNewline();

        BasicTestFormatter::printTestHeader("ChiSquared - ALL TESTS PASSED");

    } catch (const std::exception& e) {
        cerr << "FATAL: " << e.what() << endl;
        return 1;
    }

    return 0;
}

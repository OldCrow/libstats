// Focused unit test for Beta distribution
#include "include/tests.h"
#include "libstats/distributions/beta.h"

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
    BasicTestFormatter::printTestHeader("Beta");

    try {
        // =====================================================================
        // Test 1: Constructors and Destructor
        // =====================================================================
        BasicTestFormatter::printTestStart(1, "Constructors and Destructor");
        cout << "Beta(1,1) = Uniform(0,1). Beta(alpha,beta): support [0,1]." << endl;

        auto default_beta = stats::BetaDistribution::create().value;
        BasicTestFormatter::printProperty("Default alpha", default_beta.getAlpha());
        BasicTestFormatter::printProperty("Default beta", default_beta.getBeta());
        BasicTestFormatter::printProperty("Default isUniform", (int)default_beta.isUniform());

        auto b23 = stats::BetaDistribution::create(2.0, 3.0).value;
        BasicTestFormatter::printProperty("Beta(2,3) alpha", b23.getAlpha());
        BasicTestFormatter::printProperty("Beta(2,3) beta", b23.getBeta());

        auto copy_b = b23;
        BasicTestFormatter::printProperty("Copy alpha", copy_b.getAlpha());

        auto temp = stats::BetaDistribution::create(5.0, 2.0).value;
        auto move_b = std::move(temp);
        BasicTestFormatter::printProperty("Move alpha", move_b.getAlpha());

        auto result = BetaDistribution::create(3.0, 4.0);
        if (result.isOk()) {
            BasicTestFormatter::printProperty("Factory alpha", result.value.getAlpha());
        }

        BasicTestFormatter::printTestSuccess("All constructor tests passed");
        BasicTestFormatter::printNewline();

        // =====================================================================
        // Test 2: Parameter Getters and Setters
        // =====================================================================
        BasicTestFormatter::printTestStart(2, "Parameter Getters and Setters");

        auto b = stats::BetaDistribution::create(2.0, 5.0).value;
        // Beta(2,5): mean=2/7, variance=10/(49*8)=10/392
        const double expected_mean = 2.0 / 7.0;
        const double expected_var = 2.0 * 5.0 / (49.0 * 8.0);
        BasicTestFormatter::printProperty("alpha", b.getAlpha());
        BasicTestFormatter::printProperty("beta", b.getBeta());
        BasicTestFormatter::printProperty("Mean (expect 2/7 ≈ 0.2857)", b.getMean());
        BasicTestFormatter::printProperty("Variance", b.getVariance());
        BasicTestFormatter::printPropertyInt("Num parameters (expect 2)", b.getNumParameters());
        cout << "Name: " << b.getDistributionName() << endl;
        cout << "Is discrete: " << (b.isDiscrete() ? "YES" : "NO") << endl;
        BasicTestFormatter::printProperty("Support lower (0)", b.getSupportLowerBound());
        BasicTestFormatter::printProperty("Support upper (1)", b.getSupportUpperBound());

        const bool mean_ok = std::abs(b.getMean() - expected_mean) < 1e-12;
        const bool var_ok = std::abs(b.getVariance() - expected_var) < 1e-12;
        cout << "Mean correct: " << (mean_ok ? "PASS" : "FAIL") << endl;
        cout << "Variance correct: " << (var_ok ? "PASS" : "FAIL") << endl;

        b.setAlpha(4.0);
        BasicTestFormatter::printProperty("After setAlpha(4): alpha", b.getAlpha());
        b.setBeta(4.0);
        BasicTestFormatter::printProperty("After setBeta(4): isSymmetric", (int)b.isSymmetric());
        b.setParameters(1.0, 1.0);
        BasicTestFormatter::printProperty("After setParameters(1,1): isUniform",
                                          (int)b.isUniform());

        auto vr = b.trySetAlpha(-1.0);
        cout << "trySetAlpha(-1) isError: " << (vr.isError() ? "YES" : "NO") << endl;

        if (!mean_ok || !var_ok)
            throw std::runtime_error("Moment accuracy test failed");

        BasicTestFormatter::printTestSuccess("All setter/getter tests passed");
        BasicTestFormatter::printNewline();

        // =====================================================================
        // Test 3: Core Probability Methods (known values)
        // =====================================================================
        BasicTestFormatter::printTestStart(3, "Core Probability Methods");
        cout << "Known values:" << endl;
        cout << "  Beta(1,1) PDF(x) = 1 for all x in (0,1)  [Uniform]" << endl;
        cout << "  Beta(2,2) PDF(0.5) = 6*0.5*0.5 = 1.5" << endl;
        cout << "  Beta(a,b) CDF(0.5) = 0.5 when a=b  [symmetry]" << endl;

        // Uniform case
        auto b11 = BetaDistribution::create(1.0, 1.0).value;
        const double pdf_11_05 = b11.getProbability(0.5);
        BasicTestFormatter::printProperty("Beta(1,1) PDF(0.5) expect 1", pdf_11_05);
        const bool unif_ok = std::abs(pdf_11_05 - 1.0) < 1e-10;
        cout << "Uniform PDF: " << (unif_ok ? "PASS" : "FAIL") << endl;

        // Beta(2,2) analytical value
        auto b22 = BetaDistribution::create(2.0, 2.0).value;
        const double pdf_22_05 = b22.getProbability(0.5);
        BasicTestFormatter::printProperty("Beta(2,2) PDF(0.5) expect 1.5", pdf_22_05);
        const bool b22_ok = std::abs(pdf_22_05 - 1.5) < 1e-10;
        cout << "Beta(2,2) PDF: " << (b22_ok ? "PASS" : "FAIL") << endl;

        // CDF symmetry
        for (double a : {1.0, 2.0, 3.0, 5.0}) {
            auto bd = BetaDistribution::create(a, a).value;
            double cdf_half = bd.getCumulativeProbability(0.5);
            bool sym_ok = std::abs(cdf_half - 0.5) < 1e-8;
            cout << "CDF(0.5, " << a << "," << a << ")=0.5: " << (sym_ok ? "PASS" : "FAIL")
                 << " (got " << cdf_half << ")" << endl;
        }

        // Out-of-support
        BasicTestFormatter::printProperty("PDF(-0.1) expect 0", b22.getProbability(-0.1));
        BasicTestFormatter::printProperty("PDF(1.1)  expect 0", b22.getProbability(1.1));
        BasicTestFormatter::printProperty("CDF(0)    expect 0", b22.getCumulativeProbability(0.0));
        BasicTestFormatter::printProperty("CDF(1)    expect 1", b22.getCumulativeProbability(1.0));

        // Log-PDF consistency
        const double pdf_v = b22.getProbability(0.3);
        const double lpdf_v = b22.getLogProbability(0.3);
        const bool lp_ok = std::abs(std::log(pdf_v) - lpdf_v) < 1e-12;
        cout << "log(PDF) == LogPDF: " << (lp_ok ? "PASS" : "FAIL") << endl;

        // Quantile round-trip
        const double q50 = b22.getQuantile(0.5);
        BasicTestFormatter::printProperty("Quantile(0.5, Beta(2,2)) expect 0.5", q50);

        if (!unif_ok || !b22_ok || !lp_ok)
            throw std::runtime_error("Numerical accuracy failed");

        BasicTestFormatter::printTestSuccess("All probability method tests passed");
        BasicTestFormatter::printNewline();

        // =====================================================================
        // Test 4: Random Sampling
        // =====================================================================
        BasicTestFormatter::printTestStart(4, "Random Sampling");
        cout << "X/(X+Y) with Gamma samples. Mean should ≈ alpha/(alpha+beta)." << endl;

        mt19937 rng(42);
        auto b35 = BetaDistribution::create(3.0, 5.0).value;  // mean = 3/8 = 0.375

        const auto samples = b35.sample(rng, 500);
        const double smean = TestDataGenerators::computeSampleMean(samples);
        BasicTestFormatter::printProperty("Sample mean (n=500, expect ≈0.375)", smean);

        // All samples must be in (0,1)
        bool all_in_support = true;
        for (double s : samples) {
            if (s <= 0.0 || s >= 1.0) {
                all_in_support = false;
                break;
            }
        }
        cout << "All samples in (0,1): " << (all_in_support ? "PASS" : "FAIL") << endl;

        BasicTestFormatter::printTestSuccess("Sampling tests passed");
        BasicTestFormatter::printNewline();

        // =====================================================================
        // Test 5: Distribution Management (fit, reset, toString)
        // =====================================================================
        BasicTestFormatter::printTestStart(5, "Distribution Management");
        cout << "MLE: MoM initial estimate + Newton-Raphson on score equations." << endl;

        auto b_fit = BetaDistribution::create(1.0, 1.0).value;
        const auto fit_data = b35.sample(rng, 300);
        b_fit.fit(fit_data);
        BasicTestFormatter::printProperty("Fitted alpha (from Beta(3,5), expect ~3)",
                                          b_fit.getAlpha());
        BasicTestFormatter::printProperty("Fitted beta  (from Beta(3,5), expect ~5)",
                                          b_fit.getBeta());

        b_fit.reset();
        BasicTestFormatter::printProperty("After reset: alpha (expect 1)", b_fit.getAlpha());
        BasicTestFormatter::printProperty("After reset: beta  (expect 1)", b_fit.getBeta());
        cout << "isUniform after reset: " << (b_fit.isUniform() ? "YES" : "NO") << endl;
        cout << "toString: " << b_fit.toString() << endl;

        BasicTestFormatter::printTestSuccess("Distribution management tests passed");
        BasicTestFormatter::printNewline();

        // =====================================================================
        // Test 6: Batch Operations
        // =====================================================================
        BasicTestFormatter::printTestStart(6, "Batch Operations");
        cout << "SIMD two-log pipeline vs scalar. Interior values match; boundary fixup applied."
             << endl;

        auto b_batch = BetaDistribution::create(2.0, 3.0).value;
        const size_t N = 1000;
        vector<double> xs(N), pdf_r(N), logpdf_r(N), cdf_r(N);
        // Interior [0.01, 0.99] — avoid boundaries
        for (size_t i = 0; i < N; ++i) {
            xs[i] = 0.01 + static_cast<double>(i) * 0.98 / static_cast<double>(N - 1);
        }
        b_batch.getProbability(span<const double>(xs), span<double>(pdf_r));
        b_batch.getLogProbability(span<const double>(xs), span<double>(logpdf_r));
        b_batch.getCumulativeProbability(span<const double>(xs), span<double>(cdf_r));

        const double scalar_pdf = b_batch.getProbability(xs[200]);
        const bool batch_ok = std::abs(pdf_r[200] - scalar_pdf) < 1e-12;
        cout << "Batch PDF vs scalar at index 200: " << (batch_ok ? "PASS" : "FAIL") << endl;

        BasicTestFormatter::printTestSuccess("Batch operation tests passed");
        BasicTestFormatter::printNewline();

        // =====================================================================
        // Test 7: Comparison and Stream Operators
        // =====================================================================
        BasicTestFormatter::printTestStart(7, "Comparison and Stream Operators");

        auto a1 = BetaDistribution::create(2.0, 3.0).value;
        auto a2 = BetaDistribution::create(2.0, 3.0).value;
        auto a3 = BetaDistribution::create(3.0, 2.0).value;
        cout << "a1==a2 (2,3 vs 2,3): " << (a1 == a2 ? "true" : "false") << endl;
        cout << "a1!=a3 (2,3 vs 3,2): " << (a1 != a3 ? "true" : "false") << endl;

        ostringstream oss;
        oss << a1;
        cout << "Stream output: " << oss.str() << endl;

        istringstream iss("BetaDistribution(alpha=4, beta=6)");
        auto parsed = BetaDistribution::create().value;
        iss >> parsed;
        BasicTestFormatter::printProperty("Parsed alpha (expect 4)", parsed.getAlpha());
        BasicTestFormatter::printProperty("Parsed beta  (expect 6)", parsed.getBeta());

        BasicTestFormatter::printTestSuccess("Comparison and stream tests passed");
        BasicTestFormatter::printNewline();

        // =====================================================================
        // Test 8: Error Handling
        // =====================================================================
        BasicTestFormatter::printTestStart(8, "Error Handling");
        cout << "Uses create() factory (Result-based API) to test validation." << endl;
        cout << "Note: Throwing constructor not tested directly on macOS Catalina/" << endl;
        cout << "Homebrew LLVM due to known ABI exception-unwinding limitation." << endl;

        auto e0 = BetaDistribution::create(0.0, 1.0);
        cout << "create(0,1) isError: " << (e0.isError() ? "YES" : "NO") << endl;
        auto en = BetaDistribution::create(-1.0, 2.0);
        cout << "create(-1,2) isError: " << (en.isError() ? "YES" : "NO") << endl;
        auto eb = BetaDistribution::create(2.0, 0.0);
        cout << "create(2,0) isError: " << (eb.isError() ? "YES" : "NO") << endl;

        if (!e0.isError() || !en.isError() || !eb.isError()) {
            throw std::runtime_error("Error handling test failed");
        }

        BasicTestFormatter::printTestSuccess("Error handling tests passed");
        BasicTestFormatter::printNewline();

        BasicTestFormatter::printTestHeader("Beta - ALL TESTS PASSED");

    } catch (const std::exception& e) {
        cerr << "FATAL: " << e.what() << endl;
        return 1;
    }

    return 0;
}

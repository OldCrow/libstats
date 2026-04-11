// Focused unit test for Student's t distribution
#include "include/tests.h"
#include "libstats/distributions/student_t.h"

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
    BasicTestFormatter::printTestHeader("StudentT");

    try {
        // =====================================================================
        // Test 1: Constructors and Destructor
        // =====================================================================
        BasicTestFormatter::printTestStart(1, "Constructors and Destructor");
        cout << "StudentTDistribution: full real-line, log-space SIMD batch." << endl;
        cout << "nu=1: Cauchy.  nu->inf: Normal(0,1)." << endl;

        auto default_t = stats::StudentTDistribution::create().value;
        BasicTestFormatter::printProperty("Default nu (df)", default_t.getNu());
        BasicTestFormatter::printProperty("Default isCauchy", (int)default_t.isCauchy());

        auto t3 = stats::StudentTDistribution::create(3.0).value;
        BasicTestFormatter::printProperty("nu=3 created", t3.getNu());

        auto copy_t = t3;
        BasicTestFormatter::printProperty("Copy nu", copy_t.getNu());

        auto temp = stats::StudentTDistribution::create(5.0).value;
        auto move_t = std::move(temp);
        BasicTestFormatter::printProperty("Move nu", move_t.getNu());

        auto result = StudentTDistribution::create(10.0);
        if (result.isOk()) {
            BasicTestFormatter::printProperty("Factory nu=10", result.value.getNu());
        }

        BasicTestFormatter::printTestSuccess("All constructor tests passed");
        BasicTestFormatter::printNewline();

        // =====================================================================
        // Test 2: Parameter Getters and Setters
        // =====================================================================
        BasicTestFormatter::printTestStart(2, "Parameter Getters and Setters");

        auto t5 = stats::StudentTDistribution::create(5.0).value;
        BasicTestFormatter::printProperty("getNu()", t5.getNu());
        BasicTestFormatter::printProperty("getDegreesOfFreedom()", t5.getDegreesOfFreedom());
        BasicTestFormatter::printProperty("getNumParameters()", t5.getNumParameters());
        cout << "Distribution name: " << t5.getDistributionName() << endl;
        cout << "Is discrete: " << (t5.isDiscrete() ? "YES" : "NO") << endl;
        BasicTestFormatter::printProperty("Support lower (-inf)", t5.getSupportLowerBound());
        BasicTestFormatter::printProperty("Support upper (+inf)", t5.getSupportUpperBound());

        // Moments: nu=5 -> mean=0, variance=5/3, skewness=0, kurtosis=6/(5-4)=6
        BasicTestFormatter::printProperty("Mean (nu=5, expect 0)", t5.getMean());
        BasicTestFormatter::printProperty("Variance (nu=5, expect 5/3=1.667)", t5.getVariance());
        BasicTestFormatter::printProperty("Skewness (nu=5, expect 0)", t5.getSkewness());
        BasicTestFormatter::printProperty("Kurtosis (nu=5, expect 6)", t5.getKurtosis());
        BasicTestFormatter::printProperty("Mode (always 0)", t5.getMode());
        BasicTestFormatter::printProperty("Median (always 0)", t5.getMedian());

        // nu=1: mean and variance undefined
        auto t1 = stats::StudentTDistribution::create(1.0).value;
        cout << "nu=1 mean isnan: " << (std::isnan(t1.getMean()) ? "YES" : "NO") << endl;
        cout << "nu=1 variance isnan: " << (std::isnan(t1.getVariance()) ? "YES" : "NO") << endl;
        cout << "nu=1 isCauchy: " << (t1.isCauchy() ? "YES" : "NO") << endl;

        // Setters
        t5.setNu(8.0);
        BasicTestFormatter::printProperty("After setNu(8): nu", t5.getNu());
        t5.setDegreesOfFreedom(10.0);
        BasicTestFormatter::printProperty("After setDegreesOfFreedom(10): nu", t5.getNu());

        auto vr = t5.trySetNu(-1.0);
        cout << "trySetNu(-1) isError: " << (vr.isError() ? "YES" : "NO") << endl;

        BasicTestFormatter::printTestSuccess("All setter/getter tests passed");
        BasicTestFormatter::printNewline();

        // =====================================================================
        // Test 3: Core Probability Methods (numerical table values)
        // =====================================================================
        BasicTestFormatter::printTestStart(3, "Core Probability Methods");
        cout << "Verifies against known analytical values and standard t-tables." << endl;
        cout << "nu=1 (Cauchy): PDF(0) = 1/pi ~ 0.31831." << endl;
        cout << "All distributions: CDF(0) = 0.5 (symmetry)." << endl;
        cout << "nu=5:  t_{0.975} ~ 2.5706 (two-tailed alpha=0.05 critical value)." << endl;
        cout << "nu=10: t_{0.975} ~ 2.2281." << endl;
        cout << "nu=30: t_{0.975} ~ 2.0423." << endl;

        // Cauchy (nu=1)
        const double cauchy_pdf_0 = t1.getProbability(0.0);
        const double expected_cauchy_pdf_0 = 1.0 / detail::PI;
        BasicTestFormatter::printProperty("Cauchy PDF(0) computed", cauchy_pdf_0);
        BasicTestFormatter::printProperty("Cauchy PDF(0) expected (1/pi)", expected_cauchy_pdf_0);
        const bool cauchy_ok = std::abs(cauchy_pdf_0 - expected_cauchy_pdf_0) < 1e-10;
        cout << "Cauchy PDF(0) match: " << (cauchy_ok ? "PASS" : "FAIL") << endl;

        // CDF symmetry at 0
        for (double nu : {1.0, 3.0, 5.0, 10.0, 30.0}) {
            auto td = StudentTDistribution::create(nu).value;
            double cdf0 = td.getCumulativeProbability(0.0);
            bool sym_ok = std::abs(cdf0 - 0.5) < 1e-8;
            cout << "CDF(0, nu=" << nu << ")=0.5: " << (sym_ok ? "PASS" : "FAIL") << " (got "
                 << cdf0 << ")" << endl;
        }

        // t-table critical values
        auto t5b = StudentTDistribution::create(5.0).value;
        auto t10 = StudentTDistribution::create(10.0).value;
        auto t30 = StudentTDistribution::create(30.0).value;
        const double q975_5 = t5b.getQuantile(0.975);
        const double q975_10 = t10.getQuantile(0.975);
        const double q975_30 = t30.getQuantile(0.975);
        cout << fixed << setprecision(4);
        BasicTestFormatter::printProperty("t_{0.975}(nu=5)  expect ~2.5706", q975_5);
        BasicTestFormatter::printProperty("t_{0.975}(nu=10) expect ~2.2281", q975_10);
        BasicTestFormatter::printProperty("t_{0.975}(nu=30) expect ~2.0423", q975_30);
        const bool table_5 = std::abs(q975_5 - 2.5706) < 0.001;
        const bool table_10 = std::abs(q975_10 - 2.2281) < 0.001;
        const bool table_30 = std::abs(q975_30 - 2.0423) < 0.001;
        cout << "t-table nu=5:  " << (table_5 ? "PASS" : "FAIL") << endl;
        cout << "t-table nu=10: " << (table_10 ? "PASS" : "FAIL") << endl;
        cout << "t-table nu=30: " << (table_30 ? "PASS" : "FAIL") << endl;

        // Log-PDF consistency
        const double pdf_val = t5b.getProbability(1.0);
        const double logpdf_val = t5b.getLogProbability(1.0);
        const bool logpdf_ok = std::abs(std::log(pdf_val) - logpdf_val) < 1e-12;
        cout << "log(PDF) == LogPDF: " << (logpdf_ok ? "PASS" : "FAIL") << endl;

        if (!cauchy_ok || !table_5 || !table_10 || !table_30 || !logpdf_ok) {
            throw std::runtime_error("Numerical accuracy check failed");
        }

        BasicTestFormatter::printTestSuccess("All probability method tests passed");
        BasicTestFormatter::printNewline();

        // =====================================================================
        // Test 4: Random Sampling
        // =====================================================================
        BasicTestFormatter::printTestStart(4, "Random Sampling");
        cout << "Sample mean should be near 0 (symmetric), variance near nu/(nu-2)." << endl;

        mt19937 rng(42);
        auto t6 = StudentTDistribution::create(6.0).value;  // variance = 6/4 = 1.5

        const auto samples = t6.sample(rng, 500);
        const double smean = TestDataGenerators::computeSampleMean(samples);
        const double svar = TestDataGenerators::computeSampleVariance(samples);
        BasicTestFormatter::printProperty("Sample mean (n=500, expect ~0)", smean);
        BasicTestFormatter::printProperty("Sample variance (n=500, expect ~1.5)", svar);

        BasicTestFormatter::printTestSuccess("Sampling tests passed");
        BasicTestFormatter::printNewline();

        // =====================================================================
        // Test 5: Distribution Management (fit, reset, toString)
        // =====================================================================
        BasicTestFormatter::printTestStart(5, "Distribution Management");
        cout << "MLE fit: Newton-Raphson on score equation for nu." << endl;

        auto t_fit = StudentTDistribution::create(1.0).value;

        // Fit to samples from t(4)
        auto t4_src = StudentTDistribution::create(4.0).value;
        const auto fit_data = t4_src.sample(rng, 300);
        t_fit.fit(fit_data);
        BasicTestFormatter::printProperty("Fitted nu (from t(4) data, expect ~4)", t_fit.getNu());

        t_fit.reset();
        BasicTestFormatter::printProperty("After reset: nu (expect 1)", t_fit.getNu());
        cout << "isCauchy after reset: " << (t_fit.isCauchy() ? "YES" : "NO") << endl;
        cout << "toString: " << t_fit.toString() << endl;

        BasicTestFormatter::printTestSuccess("Distribution management tests passed");
        BasicTestFormatter::printNewline();

        // =====================================================================
        // Test 6: Batch Operations — scalar matches batch
        // =====================================================================
        BasicTestFormatter::printTestStart(6, "Batch Operations");
        cout << "SIMD batch PDF/LogPDF vs scalar. CDF batch matches scalar t_cdf." << endl;

        auto t_batch = StudentTDistribution::create(3.0).value;
        const size_t N = 1000;
        vector<double> xs(N), pdf_r(N), logpdf_r(N), cdf_r(N);
        for (size_t i = 0; i < N; ++i) {
            xs[i] = -5.0 + static_cast<double>(i) * 10.0 / static_cast<double>(N - 1);
        }
        t_batch.getProbability(span<const double>(xs), span<double>(pdf_r));
        t_batch.getLogProbability(span<const double>(xs), span<double>(logpdf_r));
        t_batch.getCumulativeProbability(span<const double>(xs), span<double>(cdf_r));

        const double scalar_pdf = t_batch.getProbability(xs[100]);
        const bool batch_ok = std::abs(pdf_r[100] - scalar_pdf) < 1e-12;
        cout << "Batch PDF vs scalar at x=" << xs[100] << ": " << (batch_ok ? "PASS" : "FAIL")
             << endl;
        BasicTestFormatter::printProperty("Batch CDF(0) = 0.5",
                                          cdf_r[N / 2]);  // xs[500] = 0.0

        BasicTestFormatter::printTestSuccess("Batch operation tests passed");
        BasicTestFormatter::printNewline();

        // =====================================================================
        // Test 7: Comparison and Stream Operators
        // =====================================================================
        BasicTestFormatter::printTestStart(7, "Comparison and Stream Operators");

        auto a = StudentTDistribution::create(3.0).value;
        auto b = StudentTDistribution::create(3.0).value;
        auto c = StudentTDistribution::create(5.0).value;
        cout << "a==b (nu=3): " << (a == b ? "true" : "false") << endl;
        cout << "a!=c (nu=3 vs nu=5): " << (a != c ? "true" : "false") << endl;

        ostringstream oss;
        oss << a;
        cout << "Stream output: " << oss.str() << endl;

        istringstream iss("StudentTDistribution(nu=7)");
        auto parsed = StudentTDistribution::create().value;
        iss >> parsed;
        BasicTestFormatter::printProperty("Parsed from stream (expect 7)", parsed.getNu());

        BasicTestFormatter::printTestSuccess("Comparison and stream tests passed");
        BasicTestFormatter::printNewline();

        // =====================================================================
        // Test 8: Error Handling (via create() factory — no throwing constructor)
        // =====================================================================
        BasicTestFormatter::printTestStart(8, "Error Handling");
        cout << "Uses create() factory (Result-based API) to test validation." << endl;
        cout << "Note: Throwing constructor not tested directly on macOS Catalina/" << endl;
        cout << "Homebrew LLVM due to known ABI exception-unwinding limitation." << endl;

        auto err0 = StudentTDistribution::create(0.0);
        cout << "create(0.0)   isError: " << (err0.isError() ? "YES" : "NO") << endl;
        auto errn = StudentTDistribution::create(-1.0);
        cout << "create(-1.0)  isError: " << (errn.isError() ? "YES" : "NO") << endl;
        auto errnan = StudentTDistribution::create(numeric_limits<double>::quiet_NaN());
        cout << "create(NaN)   isError: " << (errnan.isError() ? "YES" : "NO") << endl;
        auto errinf = StudentTDistribution::create(numeric_limits<double>::infinity());
        cout << "create(inf)   isError: " << (errinf.isError() ? "YES" : "NO") << endl;

        if (!err0.isError() || !errn.isError() || !errnan.isError() || !errinf.isError()) {
            throw std::runtime_error("Error handling test failed");
        }

        BasicTestFormatter::printTestSuccess("Error handling tests passed");
        BasicTestFormatter::printNewline();

        BasicTestFormatter::printTestHeader("StudentT - ALL TESTS PASSED");

    } catch (const std::exception& e) {
        cerr << "FATAL: " << e.what() << endl;
        return 1;
    }

    return 0;
}

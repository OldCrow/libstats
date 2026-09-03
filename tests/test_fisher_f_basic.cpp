// Focused unit test for the F (Fisher-Snedecor) distribution
#include "include/basic_test_runner.h"
#include "include/tests.h"
#include "libstats/distributions/fisher_f.h"

#include <cmath>
#include <iomanip>
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

// Reference values: mpmath 1.4.1 at 50 decimal digits (see the enhanced test
// for the full table and the generating expressions), rounded to 17 significant
// digits here.
namespace {
constexpr double kF510_pdf_1 = 0.49547978348663871;
constexpr double kF510_cdf_1 = 0.53488057346219959;
constexpr double kF510_q50 = 0.93193316085104795;
}  // namespace

int main() {
    BasicTestFormatter::printTestHeader("FisherF");

    try {
        // =====================================================================
        // Test 1: Constructors and Destructor
        // =====================================================================
        BasicTestFormatter::printTestStart(1, "Constructors and Destructor");
        cout << "F(d1,d2) is a transform wrapper over Beta(d1/2, d2/2):" << endl;
        cout << "  Y ~ Beta(d1/2,d2/2)  <=>  X = (d2/d1) Y/(1-Y) ~ F(d1,d2)." << endl;

        auto default_f = stats::FDistribution::create().unwrap();
        BasicTestFormatter::printProperty("Default d1", default_f.getD1());
        BasicTestFormatter::printProperty("Default d2", default_f.getD2());

        auto f_5_10 = stats::FDistribution::create(5.0, 10.0).unwrap();
        BasicTestFormatter::printProperty("d1=5, d2=10 created; d1", f_5_10.getD1());

        auto copy_f = f_5_10;
        BasicTestFormatter::printProperty("Copy d2", copy_f.getD2());

        auto temp = stats::FDistribution::create(3.0, 7.0).unwrap();
        auto move_f = std::move(temp);
        BasicTestFormatter::printProperty("Move d2", move_f.getD2());

        auto result = FDistribution::create(4.0, 12.0);
        if (result.isOk()) {
            BasicTestFormatter::printProperty("Factory d1=4", (*result).getD1());
        }

        BasicTestFormatter::printTestSuccess("All constructor tests passed");
        BasicTestFormatter::printNewline();

        // =====================================================================
        // Test 2: Parameter Getters and Setters
        // =====================================================================
        BasicTestFormatter::printTestStart(2, "Parameter Getters and Setters");

        auto f = stats::FDistribution::create(5.0, 10.0).unwrap();

        BasicTestFormatter::printProperty("Initial d1", f.getD1());
        BasicTestFormatter::printProperty("Initial d2", f.getD2());
        BasicTestFormatter::printProperty("Mean (d2/(d2-2) = 1.25)", f.getMean());
        BasicTestFormatter::printProperty("Variance (expect 1.3541666...)", f.getVariance());
        BasicTestFormatter::printPropertyInt("Num parameters (should be 2)",
                                             f.getNumParameters());
        cout << "Distribution name: " << f.getDistributionName() << endl;
        cout << "Is discrete: " << (f.isDiscrete() ? "YES" : "NO") << endl;
        BasicTestFormatter::printProperty("Support lower", f.getSupportLowerBound());
        BasicTestFormatter::printProperty("Support upper", f.getSupportUpperBound());

        f.setD1(8.0);
        BasicTestFormatter::printProperty("After setD1(8): d1", f.getD1());

        f.setD2(20.0);
        BasicTestFormatter::printProperty("After setD2(20): d2", f.getD2());
        BasicTestFormatter::printProperty("Mean (20/18)", f.getMean());

        auto set_result = f.trySetParameters(5.0, 10.0);
        if (set_result.isOk()) {
            BasicTestFormatter::printProperty("trySetParameters(5,10): d1", f.getD1());
        }

        auto bad_result = f.trySetD1(-1.0);
        cout << "trySetD1(-1) error (expected): " << bad_result.message() << endl;

        // Undefined moments must be NaN, not a silently wrong number.
        auto f_1_1 = stats::FDistribution::create(1.0, 1.0).unwrap();
        cout << "F(1,1) mean is NaN (d2 <= 2): " << (std::isnan(f_1_1.getMean()) ? "YES" : "NO")
             << endl;
        cout << "F(1,1) variance is NaN (d2 <= 4): "
             << (std::isnan(f_1_1.getVariance()) ? "YES" : "NO") << endl;
        if (!std::isnan(f_1_1.getMean()) || !std::isnan(f_1_1.getVariance()))
            throw std::runtime_error("Undefined moments must return NaN");

        BasicTestFormatter::printTestSuccess("All setter/getter tests passed");
        BasicTestFormatter::printNewline();

        // =====================================================================
        // Test 3: Core Probability Methods (known numerical values)
        // =====================================================================
        BasicTestFormatter::printTestStart(3, "Core Probability Methods");
        cout << "F(5,10) reference values computed with mpmath (50 dps)." << endl;

        auto ref = stats::FDistribution::create(5.0, 10.0).unwrap();

        const double pdf_at_1 = ref.getProbability(1.0);
        BasicTestFormatter::printProperty("PDF(1)   computed", pdf_at_1);
        BasicTestFormatter::printProperty("PDF(1)   expected", kF510_pdf_1);
        const bool pdf_ok = std::abs(pdf_at_1 - kF510_pdf_1) < 1e-12;
        cout << "PDF(1) match: " << (pdf_ok ? "PASS" : "FAIL") << endl;

        const double cdf_at_1 = ref.getCumulativeProbability(1.0);
        BasicTestFormatter::printProperty("CDF(1)   computed", cdf_at_1);
        BasicTestFormatter::printProperty("CDF(1)   expected", kF510_cdf_1);
        const bool cdf_ok = std::abs(cdf_at_1 - kF510_cdf_1) < 1e-10;
        cout << "CDF(1) match: " << (cdf_ok ? "PASS" : "FAIL") << endl;

        // log(PDF) == LogPDF
        const bool log_ok = std::abs(std::log(pdf_at_1) - ref.getLogProbability(1.0)) < 1e-12;
        cout << "log(PDF(1)) == LogPDF(1): " << (log_ok ? "PASS" : "FAIL") << endl;

        // Out-of-support and support edge
        BasicTestFormatter::printProperty("PDF(-1) should be 0", ref.getProbability(-1.0));
        BasicTestFormatter::printProperty("CDF(0)  should be 0",
                                          ref.getCumulativeProbability(0.0));
        BasicTestFormatter::printProperty("PDF(0)  should be 0 (d1=5 > 2)",
                                          ref.getProbability(0.0));

        // ±inf contract (#103)
        const double inf = std::numeric_limits<double>::infinity();
        BasicTestFormatter::printProperty("PDF(+inf) should be 0", ref.getProbability(inf));
        BasicTestFormatter::printProperty("CDF(+inf) should be 1",
                                          ref.getCumulativeProbability(inf));

        const double q50 = ref.getQuantile(0.5);
        BasicTestFormatter::printProperty("Quantile(0.50) computed", q50);
        BasicTestFormatter::printProperty("Quantile(0.50) expected", kF510_q50);
        const bool q_ok = std::abs(q50 - kF510_q50) < 1e-9;
        cout << "Quantile(0.50) match: " << (q_ok ? "PASS" : "FAIL") << endl;

        BasicTestFormatter::printProperty("Mode ((d1-2)/d1 * d2/(d2+2))", ref.getMode());
        BasicTestFormatter::printProperty("Median (= quantile(0.5))", ref.getMedian());
        BasicTestFormatter::printProperty("Entropy", ref.getEntropy());

        // Survival function is complement-native: it stays accurate far below
        // the point where 1 - CDF(x) has collapsed to zero.
        BasicTestFormatter::printProperty("SF(1e6) (1-CDF would be 0)",
                                          ref.getSurvivalProbability(1.0e6));

        if (!pdf_ok || !cdf_ok || !log_ok || !q_ok) {
            throw std::runtime_error("Numerical accuracy check failed");
        }

        BasicTestFormatter::printTestSuccess("All probability method tests passed");
        BasicTestFormatter::printNewline();

        // =====================================================================
        // Test 4: Random Sampling
        // =====================================================================
        BasicTestFormatter::printTestStart(4, "Random Sampling");
        cout << "Samples drawn from the Beta delegate and transformed." << endl;

        mt19937 rng(42);
        auto f_samp = stats::FDistribution::create(10.0, 40.0).unwrap();  // mean = 40/38

        const double single = f_samp.sample(rng);
        BasicTestFormatter::printProperty("Single sample (F(10,40))", single);

        const auto samples = f_samp.sample(rng, 500);
        const double smean = TestDataGenerators::computeSampleMean(samples);
        BasicTestFormatter::printProperty("Sample mean (n=500, expect ~1.053)", smean);

        bool all_positive = true;
        for (double v : samples)
            if (!(v > 0.0))
                all_positive = false;
        cout << "All samples strictly positive: " << (all_positive ? "YES" : "NO") << endl;
        if (!all_positive)
            throw std::runtime_error("Sampling produced a non-positive value");

        BasicTestFormatter::printTestSuccess("Sampling tests passed");
        BasicTestFormatter::printNewline();

        // =====================================================================
        // Test 5: Distribution Management (fit, reset, toString)
        // =====================================================================
        BasicTestFormatter::printTestStart(5, "Distribution Management");
        cout << "fit() is a documented best-effort method-of-moments estimate:" << endl;
        cout << "  d2 from the mean, then d1 from the variance, both clamped." << endl;

        auto f_fit = stats::FDistribution::create(1.0, 1.0).unwrap();
        const auto fit_data = f_samp.sample(rng, 2000);
        f_fit.fit(fit_data);
        BasicTestFormatter::printProperty("Fitted d1 (from F(10,40) data)", f_fit.getD1());
        BasicTestFormatter::printProperty("Fitted d2 (from F(10,40) data)", f_fit.getD2());
        if (!(f_fit.getD1() > 0.0) || !(f_fit.getD2() > 0.0))
            throw std::runtime_error("fit() produced non-positive degrees of freedom");

        f_fit.reset();
        BasicTestFormatter::printProperty("After reset: d1 (expect 1)", f_fit.getD1());
        BasicTestFormatter::printProperty("After reset: d2 (expect 1)", f_fit.getD2());

        cout << "toString: " << f_fit.toString() << endl;

        BasicTestFormatter::printTestSuccess("Distribution management tests passed");
        BasicTestFormatter::printNewline();

        // =====================================================================
        // Test 6: Auto-dispatch Batch Operations
        // =====================================================================
        stats::tests::BasicDistConfig cfg{"FisherF",
                                          {0.1, 0.5, 1.0, 2.0, 5.0},
                                          0.05,
                                          5.0,
                                          1e-12,  // pdf_tolerance
                                          1e-12};
        auto f_batch = stats::FDistribution::create(5.0, 10.0).unwrap();
        stats::tests::runBatchTests(cfg, f_batch);

        // =====================================================================
        // Test 7: Comparison and Stream Operators
        // =====================================================================
        BasicTestFormatter::printTestStart(7, "Comparison and Stream Operators");

        auto a = stats::FDistribution::create(5.0, 10.0).unwrap();
        auto b = stats::FDistribution::create(5.0, 10.0).unwrap();
        auto c = stats::FDistribution::create(3.0, 10.0).unwrap();

        cout << "a==b (5,10 vs 5,10): " << (a == b ? "true" : "false") << endl;
        cout << "a!=c (5,10 vs 3,10): " << (a != c ? "true" : "false") << endl;

        ostringstream oss;
        oss << a;
        cout << "Stream output: " << oss.str() << endl;

        istringstream iss("FDistribution(d1=7,d2=2.5)");
        FDistribution parsed = stats::FDistribution::create().unwrap();
        iss >> parsed;
        BasicTestFormatter::printProperty("Parsed from stream: d1 (expect 7)", parsed.getD1());
        BasicTestFormatter::printProperty("Parsed from stream: d2 (expect 2.5)", parsed.getD2());
        if (std::abs(parsed.getD1() - 7.0) > 1e-10 || std::abs(parsed.getD2() - 2.5) > 1e-10)
            throw std::runtime_error("Stream round-trip failed");

        BasicTestFormatter::printTestSuccess("Comparison and stream tests passed");
        BasicTestFormatter::printNewline();

        // =====================================================================
        // Test 8: Error Handling
        // =====================================================================
        cfg.invalid_scenarios = {
            {"d1 = 0", [] { return FDistribution::create(0.0, 1.0).isError(); }},
            {"d1 negative", [] { return FDistribution::create(-1.0, 1.0).isError(); }},
            {"d2 = 0", [] { return FDistribution::create(1.0, 0.0).isError(); }},
            {"d2 negative", [] { return FDistribution::create(1.0, -3.0).isError(); }},
            {"d1 NaN",
             [] {
                 return FDistribution::create(std::numeric_limits<double>::quiet_NaN(), 1.0)
                     .isError();
             }},
            {"d2 infinite",
             [] {
                 return FDistribution::create(1.0, std::numeric_limits<double>::infinity())
                     .isError();
             }},
            {"quantile(p) out of range",
             [] {
                 auto d = FDistribution::create(5.0, 10.0).unwrap();
                 try {
                     (void)d.getQuantile(1.5);
                 } catch (const std::invalid_argument&) {
                     return true;
                 }
                 return false;
             }},
            {"batch size mismatch",
             [] {
                 auto d = FDistribution::create(5.0, 10.0).unwrap();
                 std::vector<double> in(4, 1.0), out(3);
                 try {
                     d.getProbability(std::span<const double>(in), std::span<double>(out));
                 } catch (const std::invalid_argument&) {
                     return true;
                 }
                 return false;
             }},
        };
        stats::tests::runErrorTests(cfg);

        BasicTestFormatter::printTestHeader("FisherF - ALL TESTS PASSED");

    } catch (const std::exception& e) {
        cerr << "FATAL: " << e.what() << endl;
        return 1;
    }

    return 0;
}

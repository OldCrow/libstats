// Focused unit test for the Inverse Gamma distribution
#include "include/basic_test_runner.h"
#include "include/tests.h"
#include "libstats/distributions/inverse_gamma.h"

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

// Reference values: mpmath 1.4.1 at 50 decimal digits (the enhanced test holds
// the full table and the generating expressions), rounded to 17 significant
// digits here.
namespace {
constexpr double kIG32_pdf_1 = 0.54134113294645077;
constexpr double kIG32_cdf_1 = 0.67667641618306346;
constexpr double kIG32_q50 = 0.7479262863802243;
}  // namespace

int main() {
    BasicTestFormatter::printTestHeader("InverseGamma");

    try {
        // =====================================================================
        // Test 1: Constructors and Destructor
        // =====================================================================
        BasicTestFormatter::printTestStart(1, "Constructors and Destructor");
        cout << "InverseGamma(alpha, SCALE beta) delegates to Gamma(alpha, RATE beta)" << endl;
        cout << "with beta passed AS-IS: 1/X ~ Gamma(alpha, rate beta) when" << endl;
        cout << "X ~ InvGamma(alpha, scale beta). No reciprocal is taken on beta." << endl;

        auto default_ig = stats::InverseGammaDistribution::create().unwrap();
        BasicTestFormatter::printProperty("Default alpha", default_ig.getAlpha());
        BasicTestFormatter::printProperty("Default beta (scale)", default_ig.getBeta());

        auto ig32 = stats::InverseGammaDistribution::create(3.0, 2.0).unwrap();
        BasicTestFormatter::printProperty("alpha=3, beta=2 created; alpha", ig32.getAlpha());

        auto copy_ig = ig32;
        BasicTestFormatter::printProperty("Copy beta", copy_ig.getBeta());

        auto temp = stats::InverseGammaDistribution::create(5.0, 4.0).unwrap();
        auto move_ig = std::move(temp);
        BasicTestFormatter::printProperty("Move beta", move_ig.getBeta());

        auto result = InverseGammaDistribution::create(2.0, 3.0);
        if (result.isOk()) {
            BasicTestFormatter::printProperty("Factory alpha=2", (*result).getAlpha());
        }

        BasicTestFormatter::printTestSuccess("All constructor tests passed");
        BasicTestFormatter::printNewline();

        // =====================================================================
        // Test 2: Parameter Getters and Setters
        // =====================================================================
        BasicTestFormatter::printTestStart(2, "Parameter Getters and Setters");

        auto ig = stats::InverseGammaDistribution::create(3.0, 2.0).unwrap();

        BasicTestFormatter::printProperty("Initial alpha", ig.getAlpha());
        BasicTestFormatter::printProperty("Initial beta (scale)", ig.getBeta());
        BasicTestFormatter::printProperty("Mean beta/(alpha-1) = 1", ig.getMean());
        BasicTestFormatter::printProperty("Variance (expect 1)", ig.getVariance());
        BasicTestFormatter::printPropertyInt("Num parameters (should be 2)",
                                             ig.getNumParameters());
        cout << "Distribution name: " << ig.getDistributionName() << endl;
        cout << "Is discrete: " << (ig.isDiscrete() ? "YES" : "NO") << endl;
        BasicTestFormatter::printProperty("Support lower", ig.getSupportLowerBound());
        BasicTestFormatter::printProperty("Support upper", ig.getSupportUpperBound());

        ig.setAlpha(4.0);
        BasicTestFormatter::printProperty("After setAlpha(4): alpha", ig.getAlpha());

        ig.setBeta(6.0);
        BasicTestFormatter::printProperty("After setBeta(6): beta", ig.getBeta());
        BasicTestFormatter::printProperty("Mean (6/3 = 2)", ig.getMean());

        auto set_result = ig.trySetParameters(3.0, 2.0);
        if (set_result.isOk()) {
            BasicTestFormatter::printProperty("trySetParameters(3,2): alpha", ig.getAlpha());
        }

        auto bad_result = ig.trySetAlpha(-1.0);
        cout << "trySetAlpha(-1) error (expected): " << bad_result.message() << endl;

        // Undefined moments must be NaN, not a silently wrong number.
        auto ig_flat = stats::InverseGammaDistribution::create(1.0, 1.0).unwrap();
        cout << "InvGamma(1,1) mean is NaN (alpha <= 1): "
             << (std::isnan(ig_flat.getMean()) ? "YES" : "NO") << endl;
        if (!std::isnan(ig_flat.getMean()))
            throw std::runtime_error("Undefined mean must return NaN");

        BasicTestFormatter::printTestSuccess("All setter/getter tests passed");
        BasicTestFormatter::printNewline();

        // =====================================================================
        // Test 3: Core Probability Methods (known numerical values)
        // =====================================================================
        BasicTestFormatter::printTestStart(3, "Core Probability Methods");
        cout << "InvGamma(3,2) reference values computed with mpmath (50 dps)." << endl;

        auto ref = stats::InverseGammaDistribution::create(3.0, 2.0).unwrap();

        const double pdf_at_1 = ref.getProbability(1.0);
        BasicTestFormatter::printProperty("PDF(1)   computed", pdf_at_1);
        BasicTestFormatter::printProperty("PDF(1)   expected", kIG32_pdf_1);
        const bool pdf_ok = std::abs(pdf_at_1 - kIG32_pdf_1) < 1e-12;
        cout << "PDF(1) match: " << (pdf_ok ? "PASS" : "FAIL") << endl;

        const double cdf_at_1 = ref.getCumulativeProbability(1.0);
        BasicTestFormatter::printProperty("CDF(1)   computed", cdf_at_1);
        BasicTestFormatter::printProperty("CDF(1)   expected", kIG32_cdf_1);
        const bool cdf_ok = std::abs(cdf_at_1 - kIG32_cdf_1) < 1e-10;
        cout << "CDF(1) match: " << (cdf_ok ? "PASS" : "FAIL") << endl;

        const bool log_ok = std::abs(std::log(pdf_at_1) - ref.getLogProbability(1.0)) < 1e-12;
        cout << "log(PDF(1)) == LogPDF(1): " << (log_ok ? "PASS" : "FAIL") << endl;

        // Out-of-support and support edge
        BasicTestFormatter::printProperty("PDF(-1) should be 0", ref.getProbability(-1.0));
        BasicTestFormatter::printProperty("PDF(0)  should be 0", ref.getProbability(0.0));
        BasicTestFormatter::printProperty("CDF(0)  should be 0",
                                          ref.getCumulativeProbability(0.0));

        // ±inf contract (#103) across the reciprocal transform
        const double inf = std::numeric_limits<double>::infinity();
        BasicTestFormatter::printProperty("PDF(+inf) should be 0", ref.getProbability(inf));
        BasicTestFormatter::printProperty("CDF(+inf) should be 1",
                                          ref.getCumulativeProbability(inf));

        const double q50 = ref.getQuantile(0.5);
        BasicTestFormatter::printProperty("Quantile(0.50) computed", q50);
        BasicTestFormatter::printProperty("Quantile(0.50) expected", kIG32_q50);
        const bool q_ok = std::abs(q50 - kIG32_q50) < 1e-9;
        cout << "Quantile(0.50) match: " << (q_ok ? "PASS" : "FAIL") << endl;

        BasicTestFormatter::printProperty("Mode beta/(alpha+1) = 0.5", ref.getMode());
        BasicTestFormatter::printProperty("Median (= quantile(0.5))", ref.getMedian());
        BasicTestFormatter::printProperty("Entropy", ref.getEntropy());

        // The lower tail is the one that `1 - CDF_Gamma(1/x)` would flush to
        // zero; Q(alpha, beta/x) keeps it.
        BasicTestFormatter::printProperty("CDF(0.02) (~1.9e-40, not 0)",
                                          ref.getCumulativeProbability(0.02));
        BasicTestFormatter::printProperty("SF(1000)  (~1.33e-9)",
                                          ref.getSurvivalProbability(1000.0));

        if (!pdf_ok || !cdf_ok || !log_ok || !q_ok) {
            throw std::runtime_error("Numerical accuracy check failed");
        }

        BasicTestFormatter::printTestSuccess("All probability method tests passed");
        BasicTestFormatter::printNewline();

        // =====================================================================
        // Test 4: Random Sampling
        // =====================================================================
        BasicTestFormatter::printTestStart(4, "Random Sampling");
        cout << "Samples are 1/G with G drawn from the Gamma delegate." << endl;

        mt19937 rng(42);
        auto ig_samp = stats::InverseGammaDistribution::create(5.0, 4.0).unwrap();  // mean = 1

        const double single = ig_samp.sample(rng);
        BasicTestFormatter::printProperty("Single sample (InvGamma(5,4))", single);

        const auto samples = ig_samp.sample(rng, 500);
        const double smean = TestDataGenerators::computeSampleMean(samples);
        BasicTestFormatter::printProperty("Sample mean (n=500, expect ~1.0)", smean);

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
        cout << "MLE: Gamma's estimator applied to the reciprocals; the fitted" << endl;
        cout << "Gamma RATE is lifted back unchanged as this class's SCALE." << endl;

        auto ig_fit = stats::InverseGammaDistribution::create(1.0, 1.0).unwrap();
        const auto fit_data = ig_samp.sample(rng, 2000);
        ig_fit.fit(fit_data);
        BasicTestFormatter::printProperty("Fitted alpha (expect ~5)", ig_fit.getAlpha());
        BasicTestFormatter::printProperty("Fitted beta  (expect ~4)", ig_fit.getBeta());
        if (!(ig_fit.getAlpha() > 0.0) || !(ig_fit.getBeta() > 0.0))
            throw std::runtime_error("fit() produced non-positive parameters");

        ig_fit.reset();
        BasicTestFormatter::printProperty("After reset: alpha (expect 1)", ig_fit.getAlpha());
        BasicTestFormatter::printProperty("After reset: beta (expect 1)", ig_fit.getBeta());

        cout << "toString: " << ig_fit.toString() << endl;

        BasicTestFormatter::printTestSuccess("Distribution management tests passed");
        BasicTestFormatter::printNewline();

        // =====================================================================
        // Test 6: Auto-dispatch Batch Operations
        // =====================================================================
        stats::tests::BasicDistConfig cfg{"InverseGamma",
                                          {0.2, 0.5, 1.0, 2.0, 5.0},
                                          0.1,
                                          5.0,
                                          1e-12,  // pdf_tolerance
                                          1e-12};
        auto ig_batch = stats::InverseGammaDistribution::create(3.0, 2.0).unwrap();
        stats::tests::runBatchTests(cfg, ig_batch);

        // =====================================================================
        // Test 7: Comparison and Stream Operators
        // =====================================================================
        BasicTestFormatter::printTestStart(7, "Comparison and Stream Operators");

        auto a = stats::InverseGammaDistribution::create(3.0, 2.0).unwrap();
        auto b = stats::InverseGammaDistribution::create(3.0, 2.0).unwrap();
        auto c = stats::InverseGammaDistribution::create(5.0, 2.0).unwrap();

        cout << "a==b (3,2 vs 3,2): " << (a == b ? "true" : "false") << endl;
        cout << "a!=c (3,2 vs 5,2): " << (a != c ? "true" : "false") << endl;

        ostringstream oss;
        oss << a;
        cout << "Stream output: " << oss.str() << endl;

        istringstream iss("InverseGammaDistribution(alpha=7,beta=2.5)");
        InverseGammaDistribution parsed =
            stats::InverseGammaDistribution::create().unwrap();
        iss >> parsed;
        BasicTestFormatter::printProperty("Parsed from stream: alpha (expect 7)",
                                          parsed.getAlpha());
        BasicTestFormatter::printProperty("Parsed from stream: beta (expect 2.5)",
                                          parsed.getBeta());
        if (std::abs(parsed.getAlpha() - 7.0) > 1e-10 || std::abs(parsed.getBeta() - 2.5) > 1e-10)
            throw std::runtime_error("Stream round-trip failed");

        BasicTestFormatter::printTestSuccess("Comparison and stream tests passed");
        BasicTestFormatter::printNewline();

        // =====================================================================
        // Test 8: Error Handling
        // =====================================================================
        cfg.invalid_scenarios = {
            {"alpha = 0", [] { return InverseGammaDistribution::create(0.0, 1.0).isError(); }},
            {"alpha negative",
             [] { return InverseGammaDistribution::create(-1.0, 1.0).isError(); }},
            {"beta = 0", [] { return InverseGammaDistribution::create(1.0, 0.0).isError(); }},
            {"beta negative",
             [] { return InverseGammaDistribution::create(1.0, -3.0).isError(); }},
            {"alpha NaN",
             [] {
                 return InverseGammaDistribution::create(
                            std::numeric_limits<double>::quiet_NaN(), 1.0)
                     .isError();
             }},
            {"beta infinite",
             [] {
                 return InverseGammaDistribution::create(
                            1.0, std::numeric_limits<double>::infinity())
                     .isError();
             }},
            {"quantile(p) out of range",
             [] {
                 auto d = InverseGammaDistribution::create(3.0, 2.0).unwrap();
                 try {
                     (void)d.getQuantile(-0.5);
                 } catch (const std::invalid_argument&) {
                     return true;
                 }
                 return false;
             }},
            {"batch size mismatch",
             [] {
                 auto d = InverseGammaDistribution::create(3.0, 2.0).unwrap();
                 std::vector<double> in(4, 1.0), out(3);
                 try {
                     d.getCumulativeProbability(std::span<const double>(in),
                                                std::span<double>(out));
                 } catch (const std::invalid_argument&) {
                     return true;
                 }
                 return false;
             }},
        };
        stats::tests::runErrorTests(cfg);

        BasicTestFormatter::printTestHeader("InverseGamma - ALL TESTS PASSED");

    } catch (const std::exception& e) {
        cerr << "FATAL: " << e.what() << endl;
        return 1;
    }

    return 0;
}

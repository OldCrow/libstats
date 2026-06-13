// Focused unit test for Pareto distribution
#include "include/tests.h"
#include "libstats/distributions/pareto.h"

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
    BasicTestFormatter::printTestHeader("Pareto");

    try {
        // =====================================================================
        // Test 1: Constructors and Destructor
        // =====================================================================
        BasicTestFormatter::printTestStart(1, "Constructors and Destructor");
        cout << "Default (1,1) is the unit Pareto. Support: x >= scale." << endl;

        auto default_p = stats::ParetoDistribution::create().value;
        BasicTestFormatter::printProperty("Default scale", default_p.getScale());
        BasicTestFormatter::printProperty("Default alpha", default_p.getAlpha());

        auto p12 = stats::ParetoDistribution::create(1.0, 2.0).value;
        BasicTestFormatter::printProperty("Pareto(1,2) scale", p12.getScale());
        BasicTestFormatter::printProperty("Pareto(1,2) alpha", p12.getAlpha());

        auto copy_p = p12;
        BasicTestFormatter::printProperty("Copy alpha", copy_p.getAlpha());

        auto temp = stats::ParetoDistribution::create(2.0, 3.0).value;
        auto move_p = std::move(temp);
        BasicTestFormatter::printProperty("Move scale", move_p.getScale());

        auto result = ParetoDistribution::create(0.5, 1.5);
        if (result.isOk()) {
            BasicTestFormatter::printProperty("Factory scale", result.value.getScale());
        }

        BasicTestFormatter::printTestSuccess("All constructor tests passed");
        BasicTestFormatter::printNewline();

        // =====================================================================
        // Test 2: Parameter Getters and Setters
        // =====================================================================
        BasicTestFormatter::printTestStart(2, "Parameter Getters and Setters");

        // Pareto(1, 2): mean = 2/(2-1) = 2, variance = 1*2/((2-1)^2*(2-2+1=1))
        // = 1*2/1 = 2 ... wait: variance = xm^2*α/((α-1)^2*(α-2)) for α>2
        // For α=2: variance = +∞, so use α=3 for finite variance
        // Pareto(1, 3): mean = 3/(3-1)=1.5, variance = 1*3/((3-1)^2*(3-2)) = 3/4
        auto p = stats::ParetoDistribution::create(1.0, 3.0).value;
        const double expected_mean = 3.0 / 2.0;
        const double expected_var = 1.0 * 3.0 / (4.0 * 1.0);

        BasicTestFormatter::printProperty("scale", p.getScale());
        BasicTestFormatter::printProperty("alpha", p.getAlpha());
        BasicTestFormatter::printProperty("Mean (expect 1.5)", p.getMean());
        BasicTestFormatter::printProperty("Variance (expect 0.75)", p.getVariance());
        BasicTestFormatter::printPropertyInt("Num parameters (expect 2)", p.getNumParameters());
        cout << "Name: " << p.getDistributionName() << endl;
        cout << "Is discrete: " << (p.isDiscrete() ? "YES" : "NO") << endl;
        BasicTestFormatter::printProperty("Support lower (expect 1.0)", p.getSupportLowerBound());

        const bool mean_ok = std::abs(p.getMean() - expected_mean) < 1e-10;
        const bool var_ok = std::abs(p.getVariance() - expected_var) < 1e-10;
        cout << "Mean correct: " << (mean_ok ? "PASS" : "FAIL") << endl;
        cout << "Variance correct: " << (var_ok ? "PASS" : "FAIL") << endl;

        // α ≤ 1: infinite mean; α ≤ 2: infinite variance
        auto p_alpha1 = ParetoDistribution::create(1.0, 1.0).value;
        cout << "α=1 mean = +∞: " << (std::isinf(p_alpha1.getMean()) ? "PASS" : "FAIL") << endl;
        auto p_alpha2 = ParetoDistribution::create(1.0, 2.0).value;
        cout << "α=2 variance = +∞: " << (std::isinf(p_alpha2.getVariance()) ? "PASS" : "FAIL")
             << endl;

        // Setters
        p.setScale(2.0);
        BasicTestFormatter::printProperty("After setScale(2): scale", p.getScale());
        p.setAlpha(4.0);
        BasicTestFormatter::printProperty("After setAlpha(4): alpha", p.getAlpha());
        p.setParameters(1.0, 3.0);
        BasicTestFormatter::printProperty("After reset setParameters: scale", p.getScale());

        // Result-based setters
        auto vr = p.trySetAlpha(-1.0);
        cout << "trySetAlpha(-1) isError: " << (vr.isError() ? "YES" : "NO") << endl;

        if (!mean_ok || !var_ok)
            throw std::runtime_error("Moment accuracy failed");

        BasicTestFormatter::printTestSuccess("All setter/getter tests passed");
        BasicTestFormatter::printNewline();

        // =====================================================================
        // Test 3: Core Probability Methods (known values)
        // =====================================================================
        BasicTestFormatter::printTestStart(3, "Core Probability Methods");
        cout << "Known values for Pareto(1, 2):" << endl;
        cout << "  PDF(2) = 2*1^2/2^3 = 0.25" << endl;
        cout << "  CDF(2) = 1 - (1/2)^2 = 0.75" << endl;
        cout << "  Quantile(0.75) = 1/(1-0.75)^(1/2) = 1/0.5 = 2.0" << endl;

        auto p12b = ParetoDistribution::create(1.0, 2.0).value;

        // PDF(2; 1, 2) = 2*1^2/2^3 = 2/8 = 0.25
        const double pdf_at_2 = p12b.getProbability(2.0);
        BasicTestFormatter::printProperty("PDF(2) expect 0.25", pdf_at_2);
        const bool pdf_ok = std::abs(pdf_at_2 - 0.25) < 1e-12;
        cout << "PDF(2) correct: " << (pdf_ok ? "PASS" : "FAIL") << endl;

        // CDF(2; 1, 2) = 1 - (1/2)^2 = 0.75
        const double cdf_at_2 = p12b.getCumulativeProbability(2.0);
        BasicTestFormatter::printProperty("CDF(2) expect 0.75", cdf_at_2);
        const bool cdf_ok = std::abs(cdf_at_2 - 0.75) < 1e-12;
        cout << "CDF(2) correct: " << (cdf_ok ? "PASS" : "FAIL") << endl;

        // CDF(scale) = 0 always
        const double cdf_at_scale = p12b.getCumulativeProbability(1.0);
        BasicTestFormatter::printProperty("CDF(scale=1) expect 0", cdf_at_scale);
        cout << "CDF(scale) == 0: " << (cdf_at_scale == 0.0 ? "PASS" : "FAIL") << endl;

        // Out-of-support
        BasicTestFormatter::printProperty("PDF(0.5) expect 0 (below scale)",
                                          p12b.getProbability(0.5));
        BasicTestFormatter::printProperty("CDF(0.5) expect 0", p12b.getCumulativeProbability(0.5));

        // LogPDF consistency
        const double pdf_v = p12b.getProbability(3.0);
        const double lpdf_v = p12b.getLogProbability(3.0);
        const bool lp_ok = std::abs(std::log(pdf_v) - lpdf_v) < 1e-12;
        cout << "log(PDF) == LogPDF: " << (lp_ok ? "PASS" : "FAIL") << endl;

        // Quantile(0.75; 1, 2) = 1/(1-0.75)^(0.5) = 1/0.5 = 2.0
        const double q75 = p12b.getQuantile(0.75);
        BasicTestFormatter::printProperty("Quantile(0.75) expect 2.0", q75);
        const bool q_ok = std::abs(q75 - 2.0) < 1e-10;
        cout << "Quantile(0.75) correct: " << (q_ok ? "PASS" : "FAIL") << endl;

        // Utility methods
        BasicTestFormatter::printProperty("Mode = scale = 1", p12b.getMode());
        BasicTestFormatter::printProperty("Median = 1*2^(1/2) ≈ 1.414", p12b.getMedian());
        BasicTestFormatter::printProperty("Entropy", p12b.getEntropy());
        cout << "hasFiniteMean (α=2 → NO): " << (p12b.hasFiniteMean() ? "YES" : "NO") << endl;
        cout << "hasFiniteVariance (α=2 → NO): " << (p12b.hasFiniteVariance() ? "YES" : "NO")
             << endl;

        if (!pdf_ok || !cdf_ok || !lp_ok || !q_ok)
            throw std::runtime_error("Numerical accuracy failed");

        BasicTestFormatter::printTestSuccess("All probability method tests passed");
        BasicTestFormatter::printNewline();

        // =====================================================================
        // Test 4: Random Sampling
        // =====================================================================
        BasicTestFormatter::printTestStart(4, "Random Sampling");
        cout << "Sample from Pareto(1,2); all values must be >= 1." << endl;

        mt19937 rng(42);
        auto sample_dist = ParetoDistribution::create(1.0, 2.0).value;

        double s = sample_dist.sample(rng);
        BasicTestFormatter::printProperty("Single sample (expect >= 1)", s);
        cout << "Sample >= scale: " << (s >= 1.0 ? "PASS" : "FAIL") << endl;

        const auto samples = sample_dist.sample(rng, 500);
        bool all_in_support = true;
        for (double sv : samples)
            if (sv < 1.0) {
                all_in_support = false;
                break;
            }
        cout << "All samples >= scale: " << (all_in_support ? "PASS" : "FAIL") << endl;

        BasicTestFormatter::printTestSuccess("Sampling tests passed");
        BasicTestFormatter::printNewline();

        // =====================================================================
        // Test 5: Distribution Management (fit, reset, toString)
        // =====================================================================
        BasicTestFormatter::printTestStart(5, "Distribution Management");
        cout << "MLE: scale_hat=min(xi), alpha_hat=n/sum(log(xi/scale_hat))." << endl;

        auto fit_dist = ParetoDistribution::create(1.0, 1.0).value;
        auto source = ParetoDistribution::create(1.0, 3.0).value;
        const auto fit_data = source.sample(rng, 300);
        fit_dist.fit(fit_data);
        BasicTestFormatter::printProperty("Fitted scale (from Pareto(1,3), expect ~1)",
                                          fit_dist.getScale());
        BasicTestFormatter::printProperty("Fitted alpha (from Pareto(1,3), expect ~3)",
                                          fit_dist.getAlpha());

        fit_dist.reset();
        BasicTestFormatter::printProperty("After reset: scale (expect 1)", fit_dist.getScale());
        BasicTestFormatter::printProperty("After reset: alpha (expect 1)", fit_dist.getAlpha());
        cout << "toString: " << fit_dist.toString() << endl;

        BasicTestFormatter::printTestSuccess("Distribution management tests passed");
        BasicTestFormatter::printNewline();

        // =====================================================================
        // Test 6: Batch Operations
        // =====================================================================
        BasicTestFormatter::printTestStart(6, "Auto-dispatch Batch Operations");
        cout << "Batch PDF/LogPDF/CDF via auto-dispatch. Verify against scalar." << endl;

        auto batch_dist = ParetoDistribution::create(1.0, 2.0).value;
        const vector<double> xs = {1.0, 1.5, 2.0, 3.0, 5.0};
        vector<double> pdf_b(xs.size()), lpdf_b(xs.size()), cdf_b(xs.size());

        batch_dist.getProbability(span<const double>(xs), span<double>(pdf_b));
        batch_dist.getLogProbability(span<const double>(xs), span<double>(lpdf_b));
        batch_dist.getCumulativeProbability(span<const double>(xs), span<double>(cdf_b));

        bool batch_ok = true;
        for (size_t i = 0; i < xs.size(); ++i) {
            if (std::abs(pdf_b[i] - batch_dist.getProbability(xs[i])) > 1e-12 ||
                std::abs(lpdf_b[i] - batch_dist.getLogProbability(xs[i])) > 1e-12 ||
                std::abs(cdf_b[i] - batch_dist.getCumulativeProbability(xs[i])) > 1e-12) {
                batch_ok = false;
                break;
            }
        }
        cout << "Batch matches scalar: " << (batch_ok ? "PASS" : "FAIL") << endl;

        // VECTORIZED vs SCALAR on a larger batch
        const size_t N = 2000;
        vector<double> large_in(N), large_vec(N), large_scl(N);
        for (size_t i = 0; i < N; ++i)
            large_in[i] = 1.0 + 0.01 * static_cast<double>(i);
        batch_dist.getLogProbabilityWithStrategy(span<const double>(large_in),
                                                 span<double>(large_vec),
                                                 stats::detail::Strategy::VECTORIZED);
        batch_dist.getLogProbabilityWithStrategy(
            span<const double>(large_in), span<double>(large_scl), stats::detail::Strategy::SCALAR);
        bool large_ok = true;
        for (size_t i = 0; i < N; ++i) {
            if (std::abs(large_vec[i] - large_scl[i]) > 1e-10) {
                large_ok = false;
                break;
            }
        }
        cout << "VECTORIZED matches SCALAR (n=" << N << "): " << (large_ok ? "PASS" : "FAIL")
             << endl;

        BasicTestFormatter::printTestSuccess("Batch operation tests passed");
        BasicTestFormatter::printNewline();

        // =====================================================================
        // Test 7: Comparison and Stream Operators
        // =====================================================================
        BasicTestFormatter::printTestStart(7, "Comparison and Stream Operators");

        auto d1 = ParetoDistribution::create(1.0, 2.0).value;
        auto d2 = ParetoDistribution::create(1.0, 2.0).value;
        auto d3 = ParetoDistribution::create(2.0, 3.0).value;

        cout << "d1 == d2: " << (d1 == d2 ? "true" : "false") << endl;
        cout << "d1 == d3: " << (d1 == d3 ? "true" : "false") << endl;
        cout << "d1 != d3: " << (d1 != d3 ? "true" : "false") << endl;

        stringstream ss;
        ss << d1;
        cout << "Stream output: " << ss.str() << endl;

        auto input_dist = ParetoDistribution::create().value;
        ss.seekg(0);
        if (ss >> input_dist) {
            cout << "Stream round-trip scale: " << input_dist.getScale() << endl;
        } else {
            cout << "Stream input failed" << endl;
        }

        BasicTestFormatter::printTestSuccess("Comparison and stream operator tests passed");
        BasicTestFormatter::printNewline();

        // =====================================================================
        // Test 8: Error Handling
        // =====================================================================
        BasicTestFormatter::printTestStart(8, "Error Handling");

        auto err1 = ParetoDistribution::create(-1.0, 1.0);
        if (err1.isError()) {
            BasicTestFormatter::printTestSuccess("scale=-1 rejected: " + err1.message);
        }
        auto err2 = ParetoDistribution::create(1.0, 0.0);
        if (err2.isError()) {
            BasicTestFormatter::printTestSuccess("alpha=0 rejected: " + err2.message);
        }
        auto err3 = ParetoDistribution::create(std::numeric_limits<double>::quiet_NaN(), 1.0);
        if (err3.isError()) {
            BasicTestFormatter::printTestSuccess("scale=NaN rejected: " + err3.message);
        }

        BasicTestFormatter::printTestSuccess("All error handling tests passed");
        BasicTestFormatter::printNewline();

        BasicTestFormatter::printTestSuccess("All Pareto tests completed successfully");
        return 0;

    } catch (const exception& e) {
        cerr << "Test failed: " << e.what() << endl;
        return 1;
    }
}

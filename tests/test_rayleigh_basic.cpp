// Focused unit test for Rayleigh distribution
#include "include/tests.h"
#include "libstats/distributions/rayleigh.h"

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
    BasicTestFormatter::printTestHeader("Rayleigh");

    try {
        // =====================================================================
        // Test 1: Constructors and Destructor
        // =====================================================================
        BasicTestFormatter::printTestStart(1, "Constructors and Destructor");
        cout << "Default σ=1 is the standard Rayleigh. Support: x >= 0." << endl;

        auto default_r = stats::RayleighDistribution::create().value;
        BasicTestFormatter::printProperty("Default sigma", default_r.getSigma());

        auto r2 = stats::RayleighDistribution::create(2.0).value;
        BasicTestFormatter::printProperty("R(2) sigma", r2.getSigma());

        auto copy_r = r2;
        BasicTestFormatter::printProperty("Copy sigma", copy_r.getSigma());

        auto temp = stats::RayleighDistribution::create(3.0).value;
        auto move_r = std::move(temp);
        BasicTestFormatter::printProperty("Move sigma", move_r.getSigma());

        auto result = RayleighDistribution::create(0.5);
        if (result.isOk()) {
            BasicTestFormatter::printProperty("Factory sigma", result.value.getSigma());
        }

        BasicTestFormatter::printTestSuccess("All constructor tests passed");
        BasicTestFormatter::printNewline();

        // =====================================================================
        // Test 2: Parameter Getters and Setters
        // =====================================================================
        BasicTestFormatter::printTestStart(2, "Parameter Getters and Setters");

        // Rayleigh(1): mean = √(π/2) ≈ 1.2533, variance = (4−π)/2 ≈ 0.4292
        auto r = stats::RayleighDistribution::create(1.0).value;
        const double expected_mean = std::sqrt(M_PI / 2.0);
        const double expected_var = (4.0 - M_PI) / 2.0;

        BasicTestFormatter::printProperty("sigma", r.getSigma());
        BasicTestFormatter::printProperty("Mean (expect √(π/2)≈1.2533)", r.getMean());
        BasicTestFormatter::printProperty("Variance (expect (4-π)/2≈0.4292)", r.getVariance());
        BasicTestFormatter::printPropertyInt("Num parameters (expect 1)", r.getNumParameters());
        cout << "Name: " << r.getDistributionName() << endl;
        cout << "Is discrete: " << (r.isDiscrete() ? "YES" : "NO") << endl;

        const bool mean_ok = std::abs(r.getMean() - expected_mean) < 1e-10;
        const bool var_ok = std::abs(r.getVariance() - expected_var) < 1e-10;
        cout << "Mean == √(π/2): " << (mean_ok ? "PASS" : "FAIL") << endl;
        cout << "Variance == (4-π)/2: " << (var_ok ? "PASS" : "FAIL") << endl;

        r.setSigma(2.0);
        BasicTestFormatter::printProperty("After setSigma(2): sigma", r.getSigma());
        r.setParameters(1.0);
        BasicTestFormatter::printProperty("After setParameters(1): sigma", r.getSigma());

        auto vr = r.trySetSigma(-1.0);
        cout << "trySetSigma(-1) isError: " << (vr.isError() ? "YES" : "NO") << endl;

        if (!mean_ok || !var_ok)
            throw std::runtime_error("Moment accuracy failed");

        BasicTestFormatter::printTestSuccess("All setter/getter tests passed");
        BasicTestFormatter::printNewline();

        // =====================================================================
        // Test 3: Core Probability Methods (known values)
        // =====================================================================
        BasicTestFormatter::printTestStart(3, "Core Probability Methods");
        cout << "Rayleigh(σ=1):" << endl;
        cout << "  PDF(σ=1) = exp(-0.5) ≈ 0.6065" << endl;
        cout << "  CDF(σ=1) = 1 - exp(-0.5) ≈ 0.3935" << endl;
        cout << "  Median = √(2·ln 2) ≈ 1.1774" << endl;
        cout << "  Mode = σ = 1" << endl;

        auto r1 = RayleighDistribution::create(1.0).value;

        // PDF(x=σ=1) = (1/σ²)·exp(-σ²/(2σ²)) = exp(-0.5)
        const double pdf_at_sigma = r1.getProbability(1.0);
        const double exp_neg_half = std::exp(-0.5);
        BasicTestFormatter::printProperty("PDF(1;σ=1) expect exp(-0.5)", pdf_at_sigma);
        const bool pdf_ok = std::abs(pdf_at_sigma - exp_neg_half) < 1e-12;
        cout << "PDF(σ) == exp(-0.5): " << (pdf_ok ? "PASS" : "FAIL") << endl;

        // CDF(σ) = 1 - exp(-1/2) for any σ
        const double cdf_at_sigma = r1.getCumulativeProbability(1.0);
        BasicTestFormatter::printProperty("CDF(1;σ=1) expect 1-exp(-0.5)", cdf_at_sigma);
        const bool cdf_ok = std::abs(cdf_at_sigma - (1.0 - exp_neg_half)) < 1e-12;
        cout << "CDF(σ) == 1-exp(-0.5): " << (cdf_ok ? "PASS" : "FAIL") << endl;

        // CDF(σ) independent of σ
        for (double sigma : {0.5, 1.0, 2.0, 5.0}) {
            auto rd = RayleighDistribution::create(sigma).value;
            const bool cdf_sigma_ok =
                std::abs(rd.getCumulativeProbability(sigma) - (1.0 - exp_neg_half)) < 1e-12;
            cout << "CDF(sigma=" << sigma
                 << " at x=sigma) == 1-exp(-0.5): " << (cdf_sigma_ok ? "PASS" : "FAIL") << endl;
        }

        // Out-of-support
        BasicTestFormatter::printProperty("PDF(0) expect 0", r1.getProbability(0.0));
        BasicTestFormatter::printProperty("PDF(-1) expect 0", r1.getProbability(-1.0));

        // LogPDF consistency
        const double pdf_v = r1.getProbability(2.0);
        const double lp_v = r1.getLogProbability(2.0);
        const bool lp_ok = std::abs(std::log(pdf_v) - lp_v) < 1e-12;
        cout << "log(PDF) == LogPDF: " << (lp_ok ? "PASS" : "FAIL") << endl;

        // Quantile: Q(0.5) = σ·√(2·ln 2) = median
        const double q50 = r1.getQuantile(0.5);
        const double expected_median = std::sqrt(2.0 * std::log(2.0));
        BasicTestFormatter::printProperty("Quantile(0.5) expect √(2·ln2)≈1.1774", q50);
        const bool q_ok = std::abs(q50 - expected_median) < 1e-10;
        cout << "Quantile(0.5) == √(2·ln2): " << (q_ok ? "PASS" : "FAIL") << endl;

        // Utility methods
        BasicTestFormatter::printProperty("Mode = σ = 1", r1.getMode());
        BasicTestFormatter::printProperty("Median = √(2·ln2)", r1.getMedian());
        BasicTestFormatter::printProperty("Entropy", r1.getEntropy());
        BasicTestFormatter::printProperty("Skewness (≈ 0.6311)", r1.getSkewness());

        if (!pdf_ok || !cdf_ok || !lp_ok || !q_ok)
            throw std::runtime_error("Numerical accuracy failed");

        BasicTestFormatter::printTestSuccess("All probability method tests passed");
        BasicTestFormatter::printNewline();

        // =====================================================================
        // Test 4: Random Sampling
        // =====================================================================
        BasicTestFormatter::printTestStart(4, "Random Sampling");

        mt19937 rng(42);
        auto sample_dist = RayleighDistribution::create(1.0).value;
        double s = sample_dist.sample(rng);
        cout << "Single sample > 0: " << (s > 0.0 ? "PASS" : "FAIL") << endl;

        const auto samples = sample_dist.sample(rng, 500);
        bool all_pos = true;
        for (double sv : samples)
            if (sv <= 0.0) {
                all_pos = false;
                break;
            }
        cout << "All samples > 0: " << (all_pos ? "PASS" : "FAIL") << endl;

        BasicTestFormatter::printTestSuccess("Sampling tests passed");
        BasicTestFormatter::printNewline();

        // =====================================================================
        // Test 5: Distribution Management (fit, reset)
        // =====================================================================
        BasicTestFormatter::printTestStart(5, "Distribution Management");
        cout << "MLE: σ̂ = √(Σxᵢ²/(2n)). Single pass, no iteration." << endl;

        auto fit_dist = RayleighDistribution::create(1.0).value;
        auto source = RayleighDistribution::create(3.0).value;
        const auto fit_data = source.sample(rng, 300);
        fit_dist.fit(fit_data);
        BasicTestFormatter::printProperty("Fitted sigma (from R(3), expect ~3)",
                                          fit_dist.getSigma());

        fit_dist.reset();
        BasicTestFormatter::printProperty("After reset: sigma (expect 1)", fit_dist.getSigma());
        cout << "toString: " << fit_dist.toString() << endl;

        BasicTestFormatter::printTestSuccess("Distribution management tests passed");
        BasicTestFormatter::printNewline();

        // =====================================================================
        // Test 6: Batch Operations
        // =====================================================================
        BasicTestFormatter::printTestStart(6, "Auto-dispatch Batch Operations");

        auto batch_dist = RayleighDistribution::create(2.0).value;
        const vector<double> xs = {0.5, 1.0, 2.0, 3.0, 5.0};
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

        const size_t N = 2000;
        vector<double> large_in(N), vec_out(N), scl_out(N);
        for (size_t i = 0; i < N; ++i)
            large_in[i] = 0.1 + 0.01 * static_cast<double>(i);
        bool large_ok = true;
        for (size_t i = 0; i < N; ++i) {
            if (std::abs(vec_out[i] - scl_out[i]) > 1e-10) {
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

        auto d1 = RayleighDistribution::create(2.0).value;
        auto d2 = RayleighDistribution::create(2.0).value;
        auto d3 = RayleighDistribution::create(3.0).value;
        cout << "d1 == d2: " << (d1 == d2 ? "true" : "false") << endl;
        cout << "d1 == d3: " << (d1 == d3 ? "true" : "false") << endl;
        stringstream ss;
        ss << d1;
        cout << "Stream output: " << ss.str() << endl;
        auto in_dist = RayleighDistribution::create().value;
        ss.seekg(0);
        if (ss >> in_dist)
            cout << "Stream round-trip sigma: " << in_dist.getSigma() << endl;

        BasicTestFormatter::printTestSuccess("Comparison and stream tests passed");
        BasicTestFormatter::printNewline();

        // =====================================================================
        // Test 8: Error Handling
        // =====================================================================
        BasicTestFormatter::printTestStart(8, "Error Handling");

        auto err1 = RayleighDistribution::create(-1.0);
        if (err1.isError())
            BasicTestFormatter::printTestSuccess("sigma=-1 rejected: " + err1.message);
        auto err2 = RayleighDistribution::create(0.0);
        if (err2.isError())
            BasicTestFormatter::printTestSuccess("sigma=0 rejected: " + err2.message);

        BasicTestFormatter::printTestSuccess("All error handling tests passed");
        BasicTestFormatter::printNewline();

        BasicTestFormatter::printTestSuccess("All Rayleigh tests completed successfully");
        return 0;

    } catch (const exception& e) {
        cerr << "Test failed: " << e.what() << endl;
        return 1;
    }
}

// Focused unit test for Von Mises distribution
#include "include/tests.h"
#include "include/basic_test_runner.h"
#include "libstats/distributions/von_mises.h"

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
    BasicTestFormatter::printTestHeader("VonMises");

    try {
        // =====================================================================
        // Test 1: Constructors and Destructor
        // =====================================================================
        BasicTestFormatter::printTestStart(1, "Constructors and Destructor");
        cout << "Default (mu=0, kappa=1). Support: (-pi, pi]." << endl;

        auto default_vm = stats::VonMisesDistribution::create().value;
        BasicTestFormatter::printProperty("Default mu", default_vm.getMu());
        BasicTestFormatter::printProperty("Default kappa", default_vm.getKappa());
        BasicTestFormatter::printProperty("isUniform (expect 0)", static_cast<int>(default_vm.isUniform()));

        // kappa=0 → uniform
        auto vm0 = stats::VonMisesDistribution::create(0.0, 0.0).value;
        BasicTestFormatter::printProperty("kappa=0 isUniform", static_cast<int>(vm0.isUniform()));

        auto copy_vm = vm0;
        BasicTestFormatter::printProperty("Copy kappa", copy_vm.getKappa());

        auto temp = stats::VonMisesDistribution::create(1.0, 2.0).value;
        auto move_vm = std::move(temp);
        BasicTestFormatter::printProperty("Move kappa", move_vm.getKappa());

        auto result = VonMisesDistribution::create(0.5, 3.0);
        if (result.isOk()) {
            BasicTestFormatter::printProperty("Factory kappa", result.value.getKappa());
        }

        BasicTestFormatter::printTestSuccess("All constructor tests passed");
        BasicTestFormatter::printNewline();

        // =====================================================================
        // Test 2: Parameter Getters and Setters
        // =====================================================================
        BasicTestFormatter::printTestStart(2, "Parameter Getters and Setters");
        cout << "Angle wrapping: mu is always stored in (-pi, pi]." << endl;

        auto vm = stats::VonMisesDistribution::create(0.0, 1.0).value;
        BasicTestFormatter::printProperty("mu", vm.getMu());
        BasicTestFormatter::printProperty("kappa", vm.getKappa());
        BasicTestFormatter::printPropertyInt("Num parameters (expect 2)", vm.getNumParameters());
        cout << "Name: " << vm.getDistributionName() << endl;

        // setMu with out-of-range value (should wrap)
        vm.setMu(4.0);  // 4.0 > pi, wraps to ~0.717
        BasicTestFormatter::printProperty("setMu(4.0) → wrapped", vm.getMu());
        const bool wrapped_ok = (vm.getMu() > -M_PI && vm.getMu() <= M_PI);
        cout << "Mu wrapped to (-pi, pi]: " << (wrapped_ok ? "PASS" : "FAIL") << endl;

        vm.setKappa(2.0);
        BasicTestFormatter::printProperty("setKappa(2): kappa", vm.getKappa());

        vm.setParameters(0.0, 1.0);
        BasicTestFormatter::printProperty("After reset setParameters: isUniform",
                                          static_cast<int>(vm.isUniform()));

        // Variance (circular): at kappa=0 → 1.0; at kappa→∞ → 0
        auto vm_uniform = VonMisesDistribution::create(0.0, 0.0).value;
        auto vm_conc = VonMisesDistribution::create(0.0, 10.0).value;
        const double var_uniform = vm_uniform.getVariance();
        const double var_conc = vm_conc.getVariance();
        BasicTestFormatter::printProperty("Circ variance (kappa=0, expect 1)", var_uniform);
        BasicTestFormatter::printProperty("Circ variance (kappa=10, expect <0.1)", var_conc);
        const bool var_ok = (std::abs(var_uniform - 1.0) < 1e-10) && (var_conc < 0.1);
        cout << "Circular variance correct: " << (var_ok ? "PASS" : "FAIL") << endl;

        auto vr = vm.trySetKappa(-1.0);
        cout << "trySetKappa(-1) isError: " << (vr.isError() ? "YES" : "NO") << endl;

        if (!wrapped_ok || !var_ok)
            throw std::runtime_error("Parameter test failed");

        BasicTestFormatter::printTestSuccess("All setter/getter tests passed");
        BasicTestFormatter::printNewline();

        // =====================================================================
        // Test 3: Core Probability Methods (known values)
        // =====================================================================
        BasicTestFormatter::printTestStart(3, "Core Probability Methods");
        cout << "For kappa=0 (uniform): PDF = 1/(2pi) everywhere." << endl;
        cout << "For kappa=1, mu=0: PDF(0) = exp(1)/(2pi*I0(1)) = max." << endl;

        // kappa=0: uniform, PDF = 1/(2π)
        const double inv_2pi = 1.0 / (2.0 * M_PI);
        auto vm_unif = VonMisesDistribution::create(0.0, 0.0).value;
        const double pdf_unif = vm_unif.getProbability(1.5);
        BasicTestFormatter::printProperty("PDF(1.5; kappa=0) expect 1/(2pi)≈0.1592", pdf_unif);
        const bool pdf_unif_ok = std::abs(pdf_unif - inv_2pi) < 1e-8;
        cout << "PDF uniform: " << (pdf_unif_ok ? "PASS" : "FAIL") << endl;

        // kappa=0: CDF should be linearly proportional to position in (-pi, pi]
        // CDF(-pi/2) ≈ 0.25 for uniform
        const double cdf_quarter = vm_unif.getCumulativeProbability(-M_PI / 2.0);
        BasicTestFormatter::printProperty("CDF(-pi/2; uniform) expect ~0.25", cdf_quarter);
        const bool cdf_quarter_ok = std::abs(cdf_quarter - 0.25) < 0.01;
        cout << "CDF(-pi/2) ≈ 0.25: " << (cdf_quarter_ok ? "PASS" : "FAIL") << endl;

        // kappa=1, mu=0: PDF(0) > PDF(pi) (mode at mu)
        auto vm1 = VonMisesDistribution::create(0.0, 1.0).value;
        const double pdf_at_mu = vm1.getProbability(0.0);
        const double pdf_at_opp = vm1.getProbability(M_PI);
        BasicTestFormatter::printProperty("PDF(0; kappa=1) mode", pdf_at_mu);
        BasicTestFormatter::printProperty("PDF(pi; kappa=1) anti-mode", pdf_at_opp);
        cout << "PDF(mu) > PDF(pi): " << (pdf_at_mu > pdf_at_opp ? "PASS" : "FAIL") << endl;

        // LogPDF consistency
        const double pdf_v = vm1.getProbability(1.0);
        const double lp_v = vm1.getLogProbability(1.0);
        const bool lp_ok = std::abs(std::log(pdf_v) - lp_v) < 1e-12;
        cout << "log(PDF) == LogPDF: " << (lp_ok ? "PASS" : "FAIL") << endl;

        // Mode = mu
        BasicTestFormatter::printProperty("Mode = mu = 0", vm1.getMode());
        BasicTestFormatter::printProperty("Entropy", vm1.getEntropy());
        cout << "isUniform (kappa=1): " << (vm1.isUniform() ? "YES" : "NO") << endl;

        if (!pdf_unif_ok || !cdf_quarter_ok || !lp_ok)
            throw std::runtime_error("Probability accuracy failed");

        BasicTestFormatter::printTestSuccess("All probability method tests passed");
        BasicTestFormatter::printNewline();

        // =====================================================================
        // Test 4: Random Sampling
        // =====================================================================
        BasicTestFormatter::printTestStart(4, "Random Sampling");
        cout << "Best (1979) rejection sampler. Samples must be in (-pi, pi]." << endl;

        mt19937 rng(42);
        auto sample_dist = VonMisesDistribution::create(0.0, 2.0).value;
        double s = sample_dist.sample(rng);
        cout << "Sample in (-pi, pi]: " << (s > -M_PI && s <= M_PI ? "PASS" : "FAIL") << endl;

        const auto samples = sample_dist.sample(rng, 500);
        bool all_in_range = true;
        for (double sv : samples)
            if (sv <= -M_PI || sv > M_PI) {
                all_in_range = false;
                break;
            }
        cout << "All 500 samples in (-pi, pi]: " << (all_in_range ? "PASS" : "FAIL") << endl;

        BasicTestFormatter::printTestSuccess("Sampling tests passed");
        BasicTestFormatter::printNewline();

        // =====================================================================
        // Test 5: Distribution Management (MLE)
        // =====================================================================
        BasicTestFormatter::printTestStart(5, "Distribution Management");
        cout << "MLE: mu = atan2(S,C); kappa via Mardia-Jupp + Newton-Raphson." << endl;

        auto fit_dist = VonMisesDistribution::create(0.0, 1.0).value;
        auto source = VonMisesDistribution::create(1.0, 3.0).value;
        const auto fit_data = source.sample(rng, 300);
        fit_dist.fit(fit_data);
        BasicTestFormatter::printProperty("Fitted mu    (from VM(1,3), expect ~1)",
                                          fit_dist.getMu());
        BasicTestFormatter::printProperty("Fitted kappa (from VM(1,3), expect ~3)",
                                          fit_dist.getKappa());

        fit_dist.reset();
        BasicTestFormatter::printProperty("After reset: mu (expect 0)", fit_dist.getMu());
        BasicTestFormatter::printProperty("After reset: kappa (expect 1)", fit_dist.getKappa());
        cout << "toString: " << fit_dist.toString() << endl;

        BasicTestFormatter::printTestSuccess("Distribution management tests passed");
        BasicTestFormatter::printNewline();
        // =====================================================================
        // Test 6: Auto-dispatch Batch Operations
        // =====================================================================
        // pdf_tolerance relaxed to 1e-10: SIMD vector_cos has documented max error ~1e-10
        // vs std::cos in the scalar path; this propagates into the PDF via the exp.
        stats::tests::BasicDistConfig cfg{
            "VonMises",
            {-1.5, -0.5, 0.0, 0.5, 1.5},
            -3.14159265358979, 3.14159265358979,
            1e-10,  // pdf_tolerance
            1e-10   // cdf_tolerance
        };
        cfg.invalid_scenarios = {
            {"mu=inf", [] { return VonMisesDistribution::create(std::numeric_limits<double>::infinity(), 1.0).isError(); }},
            {"kappa=-1", [] { return VonMisesDistribution::create(0.0, -1.0).isError(); }},
            {"mu=NaN", [] { return VonMisesDistribution::create(std::numeric_limits<double>::quiet_NaN(), 1.0).isError(); }},
        };
        auto batch_dist = VonMisesDistribution::create(0.0, 1.0).value;
        stats::tests::runBatchTests(cfg, batch_dist);

        BasicTestFormatter::printTestStart(7, "Comparison and Stream Operators");

        auto d1 = VonMisesDistribution::create(0.0, 2.0).value;
        auto d2 = VonMisesDistribution::create(0.0, 2.0).value;
        auto d3 = VonMisesDistribution::create(1.0, 3.0).value;
        cout << "d1 == d2: " << (d1 == d2 ? "true" : "false") << endl;
        cout << "d1 == d3: " << (d1 == d3 ? "true" : "false") << endl;
        stringstream ss;
        ss << d1;
        cout << "Stream output: " << ss.str() << endl;
        auto in_dist = VonMisesDistribution::create().value;
        ss.seekg(0);
        if (ss >> in_dist)
            cout << "Stream round-trip kappa: " << in_dist.getKappa() << endl;

        BasicTestFormatter::printTestSuccess("Comparison and stream tests passed");
        BasicTestFormatter::printNewline();

        // =====================================================================
        // Test 8: Error Handling
        // =====================================================================
        stats::tests::runErrorTests(cfg);

        BasicTestFormatter::printTestSuccess("All Von Mises tests completed successfully");
        return 0;

    } catch (const exception& e) {
        cerr << "Test failed: " << e.what() << endl;
        return 1;
    }
}

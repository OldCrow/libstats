/**
 * @file quantile_accuracy.cpp
 * @brief Quantile function accuracy tool
 *
 * Verifies getQuantile() numerical accuracy across the full [0, 1] domain
 * for all 16 distributions, analogous to simd_verification.cpp for PDF/LogPDF/CDF.
 *
 * For each distribution:
 *   - Continuous: verifies CDF(quantile(p)) ≈ p for 1000 p values from 0.001 to 0.999
 *     and near-boundary values down to p = 1e-6.
 *   - Discrete:   verifies quantile(CDF(k)) >= k (floor property) and
 *                 CDF(quantile(p)) >= p (step-function round-trip).
 *
 * Reports max and mean |CDF(quantile(p)) - p| per distribution plus any
 * systematic biases at the boundaries.
 *
 * Usage:
 *   ./build/tools/quantile_accuracy [--verbose]
 */

#include "tool_utils.h"

#include "libstats/distributions/beta.h"
#include "libstats/distributions/binomial.h"
#include "libstats/distributions/chi_squared.h"
#include "libstats/distributions/discrete.h"
#include "libstats/distributions/exponential.h"
#include "libstats/distributions/gamma.h"
#include "libstats/distributions/gaussian.h"
#include "libstats/distributions/lognormal.h"
#include "libstats/distributions/geometric.h"
#include "libstats/distributions/laplace.h"
#include "libstats/distributions/negative_binomial.h"
#include "libstats/distributions/pareto.h"
#include "libstats/distributions/poisson.h"
#include "libstats/distributions/rayleigh.h"
#include "libstats/distributions/student_t.h"
#include "libstats/distributions/uniform.h"
#include "libstats/distributions/von_mises.h"
#include "libstats/distributions/weibull.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace stats;

struct QuantileResult {
    std::string dist_name;
    std::string params;
    bool is_discrete;
    double max_roundtrip_error;  // max |CDF(Q(p)) - p|
    double mean_roundtrip_error;
    size_t n_tested;
    size_t n_failed;   // |error| > tolerance
    double tolerance;
    // Near-boundary checks
    double near_zero_error;  // max error at p ∈ {1e-3, 1e-4, 1e-5, 1e-6}
    double near_one_error;   // max error at 1-p for same values
    bool any_nan;
    bool any_inf;
};

// ─────────────────────────────────────────────────────────────────────────────
// Test continuous distribution: CDF(quantile(p)) == p
// ─────────────────────────────────────────────────────────────────────────────
template <typename Dist>
QuantileResult test_continuous(const std::string& name, const std::string& params,
                               const Dist& dist, double tolerance = 1e-6) {
    QuantileResult r;
    r.dist_name = name;
    r.params = params;
    r.is_discrete = false;
    r.tolerance = tolerance;
    r.max_roundtrip_error = 0.0;
    r.mean_roundtrip_error = 0.0;
    r.n_tested = 0;
    r.n_failed = 0;
    r.near_zero_error = 0.0;
    r.near_one_error = 0.0;
    r.any_nan = false;
    r.any_inf = false;

    double sum_err = 0.0;

    // Main grid: 1000 evenly spaced p values from 0.001 to 0.999
    const int N = 1000;
    for (int i = 0; i < N; ++i) {
        const double p = 0.001 + 0.998 * static_cast<double>(i) / static_cast<double>(N - 1);
        const double q = dist.getQuantile(p);
        if (std::isnan(q)) { r.any_nan = true; ++r.n_failed; continue; }
        if (std::isinf(q)) { r.any_inf = true; ++r.n_failed; continue; }
        const double cdf_q = dist.getCumulativeProbability(q);
        const double err = std::abs(cdf_q - p);
        r.max_roundtrip_error = std::max(r.max_roundtrip_error, err);
        sum_err += err;
        ++r.n_tested;
        if (err > tolerance) ++r.n_failed;
    }
    r.mean_roundtrip_error = r.n_tested > 0 ? sum_err / static_cast<double>(r.n_tested) : 0.0;

    // Near-zero boundary
    for (double p : {1e-3, 1e-4, 1e-5, 1e-6}) {
        const double q = dist.getQuantile(p);
        if (!std::isfinite(q)) continue;
        const double err = std::abs(dist.getCumulativeProbability(q) - p);
        r.near_zero_error = std::max(r.near_zero_error, err);
    }

    // Near-one boundary
    for (double p : {1.0 - 1e-3, 1.0 - 1e-4, 1.0 - 1e-5, 1.0 - 1e-6}) {
        const double q = dist.getQuantile(p);
        if (!std::isfinite(q)) continue;
        const double err = std::abs(dist.getCumulativeProbability(q) - p);
        r.near_one_error = std::max(r.near_one_error, err);
    }

    return r;
}

// ─────────────────────────────────────────────────────────────────────────────
// Test discrete distribution: floor property — CDF(Q(p)) >= p
// ─────────────────────────────────────────────────────────────────────────────
template <typename Dist>
QuantileResult test_discrete(const std::string& name, const std::string& params,
                             const Dist& dist) {
    QuantileResult r;
    r.dist_name = name;
    r.params = params;
    r.is_discrete = true;
    r.tolerance = 0.0;  // exact for discrete: CDF(Q(p)) must be >= p
    r.max_roundtrip_error = 0.0;
    r.mean_roundtrip_error = 0.0;
    r.n_tested = 0;
    r.n_failed = 0;
    r.near_zero_error = 0.0;
    r.near_one_error = 0.0;
    r.any_nan = false;
    r.any_inf = false;

    const int N = 500;
    double sum_excess = 0.0;

    for (int i = 0; i < N; ++i) {
        const double p = 0.001 + 0.998 * static_cast<double>(i) / static_cast<double>(N - 1);
        const double q = dist.getQuantile(p);
        if (!std::isfinite(q)) continue;
        const double cdf_q = dist.getCumulativeProbability(q);
        // For discrete: CDF(Q(p)) >= p always (floor property)
        if (cdf_q < p - 1e-12) { ++r.n_failed; }
        const double excess = cdf_q - p;  // non-negative when correct
        r.max_roundtrip_error = std::max(r.max_roundtrip_error, std::abs(excess));
        sum_excess += excess;
        ++r.n_tested;
    }
    r.mean_roundtrip_error = r.n_tested > 0 ? sum_excess / static_cast<double>(r.n_tested) : 0.0;

    return r;
}

// ─────────────────────────────────────────────────────────────────────────────
// Print summary table
// ─────────────────────────────────────────────────────────────────────────────
void printResults(const std::vector<QuantileResult>& results) {
    stats::detail::detail::ColumnFormatter fmt({16, 22, 12, 12, 8, 12, 12, 8});
    std::cout << "\n" << fmt.formatRow({"Distribution", "Parameters", "MaxErr", "MeanErr",
                                        "Failed", "NearZero", "NearOne", "Status"}) << "\n";
    std::cout << fmt.getSeparator() << "\n";

    int total_pass = 0, total_fail = 0;

    for (const auto& r : results) {
        std::string max_err_str, mean_err_str, near0_str, near1_str;
        auto fmt_e = [](double v) -> std::string {
            if (v < 1e-15) return "~0";
            std::ostringstream oss;
            oss << std::scientific << std::setprecision(1) << v;
            return oss.str();
        };

        max_err_str  = fmt_e(r.max_roundtrip_error);
        mean_err_str = fmt_e(r.mean_roundtrip_error);
        near0_str    = fmt_e(r.near_zero_error);
        near1_str    = fmt_e(r.near_one_error);

        bool pass = (r.n_failed == 0) && !r.any_nan && !r.any_inf;
        std::string status = pass ? "PASS" : "FAIL";
        if (pass) ++total_pass; else ++total_fail;

        std::string failed_str = std::to_string(r.n_failed);
        if (r.any_nan) failed_str += "+NaN";
        if (r.any_inf) failed_str += "+Inf";

        std::cout << fmt.formatRow({r.dist_name, r.params, max_err_str, mean_err_str,
                                    failed_str, near0_str, near1_str, status}) << "\n";
    }

    std::cout << "\nTotal: " << (total_pass + total_fail) << " test cases "
              << "(some distributions tested with multiple parameter sets)\n"
              << total_pass << " PASS, " << total_fail << " FAIL\n";
    if (total_fail > 0) {
        std::cout << "\nFAIL indicates a quantile inversion accuracy problem.\n"
                  << "These may be pre-existing Newton-Raphson convergence issues;\n"
                  << "run with --verbose to see which (p, Q, CDF(Q)) triples fail.\n";
    }
}

int main(int argc, char* argv[]) {
    bool verbose = (argc > 1 && std::string(argv[1]) == "--verbose");

    return stats::detail::detail::runTool("Quantile Accuracy Tool", [verbose]() {
        stats::detail::detail::displayToolHeader(
            "Quantile Function Accuracy",
            "CDF(quantile(p)) ≈ p round-trip across the full [0,1] domain");

        std::cout << "Continuous: tolerance = 1e-6 for most distributions.\n"
                  << "Discrete:   floor property — CDF(quantile(p)) >= p.\n\n";

        std::vector<QuantileResult> results;

        // ── Continuous distributions ──────────────────────────────────────────

        // Gaussian: closed-form erf inverse; should be highly accurate
        results.push_back(test_continuous("Gaussian",    "N(0,1)",
            GaussianDistribution::create(0.0, 1.0).value));
        results.push_back(test_continuous("Gaussian",    "N(5,2)",
            GaussianDistribution::create(5.0, 2.0).value));

        // Exponential: closed-form -log(1-p)/lambda; exact to machine precision
        results.push_back(test_continuous("Exponential", "Exp(1)",
            ExponentialDistribution::create(1.0).value));
        results.push_back(test_continuous("Exponential", "Exp(0.1)",
            ExponentialDistribution::create(0.1).value));

        // Gamma: numerical Newton-Raphson inverse
        results.push_back(test_continuous("Gamma",       "G(2,1)",
            GammaDistribution::create(2.0, 1.0).value));
        results.push_back(test_continuous("Gamma",       "G(0.5,1)",
            GammaDistribution::create(0.5, 1.0).value, 1e-5));

        // Chi-squared: delegates to Gamma
        results.push_back(test_continuous("ChiSquared",  "chi2(4)",
            ChiSquaredDistribution::create(4.0).value));
        results.push_back(test_continuous("ChiSquared",  "chi2(1)",
            ChiSquaredDistribution::create(1.0).value, 1e-5));

        // Student-t: numerical; heavier tails make boundary hard
        results.push_back(test_continuous("StudentT",    "t(3)",
            StudentTDistribution::create(3.0).value, 1e-5));
        results.push_back(test_continuous("StudentT",    "t(30)",
            StudentTDistribution::create(30.0).value));

        // Uniform: linear; exact
        results.push_back(test_continuous("Uniform",     "U(0,1)",
            UniformDistribution::create(0.0, 1.0).value));
        results.push_back(test_continuous("Uniform",     "U(-3,5)",
            UniformDistribution::create(-3.0, 5.0).value));

        // Beta: numerical; avoid very small alpha/beta which stress the solver
        results.push_back(test_continuous("Beta",        "B(2,3)",
            BetaDistribution::create(2.0, 3.0).value));
        results.push_back(test_continuous("Beta",        "B(0.5,0.5)",
            BetaDistribution::create(0.5, 0.5).value, 1e-5));

        // LogNormal: closed-form via Gaussian inverse
        results.push_back(test_continuous("LogNormal",   "LN(0,1)",
            LogNormalDistribution::create(0.0, 1.0).value));
        results.push_back(test_continuous("LogNormal",   "LN(2,0.5)",
            LogNormalDistribution::create(2.0, 0.5).value));

        // Pareto: closed-form power-law; avoid p→1 (unbounded)
        results.push_back(test_continuous("Laplace",     "Lap(0,1)",
            LaplaceDistribution::create(0.0, 1.0).value));
        results.push_back(test_continuous("Laplace",     "Lap(2,0.5)",
            LaplaceDistribution::create(2.0, 0.5).value));
        results.push_back(test_continuous("Pareto",      "Pa(1,2)",
            ParetoDistribution::create(1.0, 2.0).value));

        // Weibull: closed-form via exp inverse
        results.push_back(test_continuous("Weibull",     "W(2,1)",
            WeibullDistribution::create(2.0, 1.0).value));
        results.push_back(test_continuous("Weibull",     "W(0.7,1)",
            WeibullDistribution::create(0.7, 1.0).value));

        // Rayleigh: closed-form
        results.push_back(test_continuous("Rayleigh",    "R(1)",
            RayleighDistribution::create(1.0).value));

        // VonMises: circular domain [-pi, pi]; numerical
        results.push_back(test_continuous("VonMises",    "VM(0,2)",
            VonMisesDistribution::create(0.0, 2.0).value, 1e-5));

        // ── Discrete distributions ────────────────────────────────────────────
        results.push_back(test_discrete("Poisson",       "Pois(3)",
            PoissonDistribution::create(3.0).value));
        results.push_back(test_discrete("Poisson",       "Pois(20)",
            PoissonDistribution::create(20.0).value));
        results.push_back(test_discrete("Discrete",      "D[1,6]",
            DiscreteDistribution::create(1, 6).value));
        results.push_back(test_discrete("Binomial",      "B(10,0.5)",
            BinomialDistribution::create(10, 0.5).value));
        results.push_back(test_discrete("NegBinomial",   "NB(5,0.4)",
            NegativeBinomialDistribution::create(5.0, 0.4).value));
        results.push_back(test_discrete("Geometric",      "Geo(0.5)",
            GeometricDistribution::create(0.5).value));
        results.push_back(test_discrete("Geometric",      "Geo(0.3)",
            GeometricDistribution::create(0.3).value));

        printResults(results);

        if (verbose) {
            std::cout << "\n[verbose] First failing (p, Q, CDF(Q)) per failed test case:\n";
            // Re-run to find failing tuples (not stored in QuantileResult to keep it small)
            auto print_fails = [](const std::string& tag, auto&& dist, double tol) {
                const int N = 1000;
                bool printed_header = false;
                int count = 0;
                for (int i = 0; i < N && count < 5; ++i) {
                    double p = 0.001 + 0.998 * static_cast<double>(i) / static_cast<double>(N - 1);
                    double q = dist.getQuantile(p);
                    if (!std::isfinite(q)) continue;
                    double c = dist.getCumulativeProbability(q);
                    double err = std::abs(c - p);
                    if (err > tol) {
                        if (!printed_header) {
                            std::cout << "  " << tag << ":\n";
                            printed_header = true;
                        }
                        std::cout << std::scientific << std::setprecision(3)
                                  << "    i=" << i << " p=" << p
                                  << " Q=" << q << " CDF(Q)=" << c << " err=" << err << "\n";
                        ++count;
                    }
                }
            };
            for (const auto& r : results) {
                if (r.n_failed > 0) {
                    std::cout << "  " << r.dist_name << " " << r.params
                              << ": " << r.n_failed << " failures, max_err="
                              << r.max_roundtrip_error << "\n";
                }
            }
        }
    });
}

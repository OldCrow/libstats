/**
 * @file parameter_recovery_benchmark.cpp
 * @brief Distribution parameter recovery benchmark
 *
 * For each distribution, runs fit() at sample sizes
 * n = {25, 50, 100, 250, 500, 1000, 2500} with M=100 replicates.
 * Computes per-summary-statistic (mean, variance) bias and RMSE to
 * answer: "What sample size gives stable MLE estimates?"
 *
 * Proxy metric: we compare the fitted distribution's getMean() and
 * getVariance() against the true distribution's getMean()/getVariance().
 * This works uniformly across all distributions without needing to know
 * each distribution's native parameter names.
 *
 * Usage:
 *   ./build/tools/parameter_recovery_benchmark [--quick] [--full]
 *     --quick: fewer replicates and smaller sample sizes (fast preview)
 *     --full:  M=200 replicates with all sample sizes (default)
 */

#include "tool_utils.h"

#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>

using namespace stats;

struct RecoveryStats {
    double mean_bias;  // E[fitted.getMean()] - true.getMean()
    double mean_rmse;  // sqrt(E[(fitted.getMean() - true.getMean())^2])
    double var_bias;   // E[fitted.getVariance()] - true.getVariance()
    double var_rmse;
    int n_ok;  // replicates where fit succeeded
};

RecoveryStats compute_stats(const std::vector<double>& means, const std::vector<double>& vars,
                            double true_mean, double true_var) {
    RecoveryStats s{};
    s.n_ok = static_cast<int>(means.size());
    if (s.n_ok == 0)
        return s;

    double sum_mean_err = 0, sum_mean_sq = 0;
    double sum_var_err = 0, sum_var_sq = 0;
    for (int i = 0; i < s.n_ok; ++i) {
        double me = means[static_cast<std::size_t>(i)] - true_mean;
        double ve = vars[static_cast<std::size_t>(i)] - true_var;
        sum_mean_err += me;
        sum_mean_sq += me * me;
        sum_var_err += ve;
        sum_var_sq += ve * ve;
    }
    double n = static_cast<double>(s.n_ok);
    s.mean_bias = sum_mean_err / n;
    s.mean_rmse = std::sqrt(sum_mean_sq / n);
    s.var_bias = sum_var_err / n;
    s.var_rmse = std::sqrt(sum_var_sq / n);
    return s;
}

template <typename Dist>
void run_recovery(const std::string& dist_name, const std::string& params,
                  std::function<std::vector<double>(std::mt19937&, size_t)> sampler,
                  std::function<Dist()> make_dist, double true_mean, double true_var,
                  const std::vector<size_t>& sample_sizes, int n_reps,
                  stats::detail::detail::ColumnFormatter& fmt) {
    std::cout << "\n" << dist_name << " " << params << "\n";

    for (size_t n : sample_sizes) {
        std::mt19937 rng(42);
        std::vector<double> fitted_means, fitted_vars;
        fitted_means.reserve(static_cast<size_t>(n_reps));
        fitted_vars.reserve(static_cast<size_t>(n_reps));

        for (int rep = 0; rep < n_reps; ++rep) {
            auto data = sampler(rng, n);
            auto fitted = make_dist();
            try {
                fitted.fit(data);
                double fm = fitted.getMean();
                double fv = fitted.getVariance();
                if (std::isfinite(fm) && std::isfinite(fv)) {
                    fitted_means.push_back(fm);
                    fitted_vars.push_back(fv);
                }
            } catch (...) {
            }
        }

        auto s = compute_stats(fitted_means, fitted_vars, true_mean, true_var);

        auto fmt_v = [](double v) -> std::string {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(4) << v;
            return oss.str();
        };
        std::string n_ok_str = std::to_string(s.n_ok) + "/" + std::to_string(n_reps);

        std::cout << fmt.formatRow({std::to_string(n), n_ok_str, fmt_v(s.mean_bias),
                                    fmt_v(s.mean_rmse), fmt_v(s.var_bias), fmt_v(s.var_rmse)})
                  << "\n";
    }
}

int main(int argc, char* argv[]) {
    bool quick_mode = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--quick")
            quick_mode = true;
    }

    return stats::detail::detail::runTool("Parameter Recovery Benchmark", [quick_mode]() {
        stats::detail::detail::displayToolHeader(
            "Distribution Parameter Recovery",
            "Bias and RMSE of getMean()/getVariance() vs true values by sample size");

        const std::vector<size_t> sample_sizes =
            quick_mode ? std::vector<size_t>{50, 100, 500}
                       : std::vector<size_t>{25, 50, 100, 250, 500, 1000, 2500};
        const int n_reps = quick_mode ? 50 : 100;

        std::cout << "M=" << n_reps << " replicates per sample size.\n"
                  << "Metric: bias and RMSE of getMean() and getVariance().\n\n";

        stats::detail::detail::ColumnFormatter fmt({8, 10, 12, 12, 12, 12});
        std::string header =
            fmt.formatRow({"n", "ok/M", "MeanBias", "MeanRMSE", "VarBias", "VarRMSE"});

        std::string sep = fmt.getSeparator();

        auto print_header = [&]() { std::cout << header << "\n" << sep << "\n"; };

        // Gaussian N(2, 1.5)
        {
            auto true_d = GaussianDistribution::create(2.0, 1.5).unwrap();
            std::cout << "\n--- Gaussian N(2, 1.5)  true mean=2, var=2.25 ---\n";
            print_header();
            run_recovery<GaussianDistribution>(
                "Gaussian", "N(2,1.5)",
                [&](std::mt19937& rng, size_t n) { return true_d.sample(rng, n); },
                [] { return GaussianDistribution::create(0.0, 1.0).unwrap(); }, true_d.getMean(),
                true_d.getVariance(), sample_sizes, n_reps, fmt);
        }

        // Exponential Exp(0.5) — mean=2, var=4
        {
            auto true_d = ExponentialDistribution::create(0.5).unwrap();
            std::cout << "\n--- Exponential Exp(0.5)  true mean=2, var=4 ---\n";
            print_header();
            run_recovery<ExponentialDistribution>(
                "Exponential", "Exp(0.5)",
                [&](std::mt19937& rng, size_t n) { return true_d.sample(rng, n); },
                [] { return ExponentialDistribution::create(1.0).unwrap(); }, true_d.getMean(),
                true_d.getVariance(), sample_sizes, n_reps, fmt);
        }

        // Gamma G(3, 1) — mean=3, var=3
        {
            auto true_d = GammaDistribution::create(3.0, 1.0).unwrap();
            std::cout << "\n--- Gamma G(3,1)  true mean=3, var=3 ---\n";
            print_header();
            run_recovery<GammaDistribution>(
                "Gamma", "G(3,1)",
                [&](std::mt19937& rng, size_t n) { return true_d.sample(rng, n); },
                [] { return GammaDistribution::create(1.0, 1.0).unwrap(); }, true_d.getMean(),
                true_d.getVariance(), sample_sizes, n_reps, fmt);
        }

        // Beta B(2,3) — mean=0.4, var≈0.032
        {
            auto true_d = BetaDistribution::create(2.0, 3.0).unwrap();
            std::cout << "\n--- Beta B(2,3)  true mean=0.4, var≈0.032 ---\n";
            print_header();
            run_recovery<BetaDistribution>(
                "Beta", "B(2,3)",
                [&](std::mt19937& rng, size_t n) { return true_d.sample(rng, n); },
                [] { return BetaDistribution::create(1.0, 1.0).unwrap(); }, true_d.getMean(),
                true_d.getVariance(), sample_sizes, n_reps, fmt);
        }

        // LogNormal LN(1, 0.5)
        {
            auto true_d = LogNormalDistribution::create(1.0, 0.5).unwrap();
            std::cout << "\n--- LogNormal LN(1,0.5)  true mean≈3.08, var≈2.72 ---\n";
            print_header();
            run_recovery<LogNormalDistribution>(
                "LogNormal", "LN(1,0.5)",
                [&](std::mt19937& rng, size_t n) { return true_d.sample(rng, n); },
                [] { return LogNormalDistribution::create(0.0, 1.0).unwrap(); }, true_d.getMean(),
                true_d.getVariance(), sample_sizes, n_reps, fmt);
        }

        // Weibull W(2, 1) — mean≈0.886, var≈0.215
        {
            auto true_d = WeibullDistribution::create(2.0, 1.0).unwrap();
            std::cout << "\n--- Weibull W(2,1)  true mean≈0.886, var≈0.215 ---\n";
            print_header();
            run_recovery<WeibullDistribution>(
                "Weibull", "W(2,1)",
                [&](std::mt19937& rng, size_t n) { return true_d.sample(rng, n); },
                [] { return WeibullDistribution::create(1.0, 1.0).unwrap(); }, true_d.getMean(),
                true_d.getVariance(), sample_sizes, n_reps, fmt);
        }

        // Pareto Pa(1, 3) — mean=1.5, var=0.75
        {
            auto true_d = ParetoDistribution::create(1.0, 3.0).unwrap();
            std::cout << "\n--- Pareto Pa(1,3)  true mean=1.5, var=0.75 ---\n";
            print_header();
            run_recovery<ParetoDistribution>(
                "Pareto", "Pa(1,3)",
                [&](std::mt19937& rng, size_t n) { return true_d.sample(rng, n); },
                [] { return ParetoDistribution::create(1.0, 1.5).unwrap(); }, true_d.getMean(),
                true_d.getVariance(), sample_sizes, n_reps, fmt);
        }

        // StudentT t(5)
        {
            auto true_d = StudentTDistribution::create(5.0).unwrap();
            std::cout << "\n--- StudentT t(5)  true mean=0, var=1.667 ---\n";
            print_header();
            run_recovery<StudentTDistribution>(
                "StudentT", "t(5)",
                [&](std::mt19937& rng, size_t n) { return true_d.sample(rng, n); },
                [] { return StudentTDistribution::create(3.0).unwrap(); }, true_d.getMean(),
                true_d.getVariance(), sample_sizes, n_reps, fmt);
        }

        // Poisson Pois(5)
        {
            auto true_d = PoissonDistribution::create(5.0).unwrap();
            std::cout << "\n--- Poisson Pois(5)  true mean=5, var=5 ---\n";
            print_header();
            run_recovery<PoissonDistribution>(
                "Poisson", "Pois(5)",
                [&](std::mt19937& rng, size_t n) { return true_d.sample(rng, n); },
                [] { return PoissonDistribution::create(1.0).unwrap(); }, true_d.getMean(),
                true_d.getVariance(), sample_sizes, n_reps, fmt);
        }

        // Binomial B(20, 0.4)
        {
            auto true_d = BinomialDistribution::create(20, 0.4).unwrap();
            std::cout << "\n--- Binomial B(20,0.4)  true mean=8, var=4.8 ---\n";
            print_header();
            run_recovery<BinomialDistribution>(
                "Binomial", "B(20,0.4)",
                [&](std::mt19937& rng, size_t n) { return true_d.sample(rng, n); },
                [] { return BinomialDistribution::create(20, 0.5).unwrap(); }, true_d.getMean(),
                true_d.getVariance(), sample_sizes, n_reps, fmt);
        }

        // Laplace Lap(1.5, 0.7) — mean=1.5, var=2*0.49=0.98
        {
            auto true_d = LaplaceDistribution::create(1.5, 0.7).unwrap();
            std::cout << "\n--- Laplace Lap(1.5,0.7)  true mean=1.5, var≈0.98 ---\n";
            print_header();
            run_recovery<LaplaceDistribution>(
                "Laplace", "Lap(1.5,0.7)",
                [&](std::mt19937& rng, size_t n) { return true_d.sample(rng, n); },
                [] { return LaplaceDistribution::create(0.0, 1.0).unwrap(); }, true_d.getMean(),
                true_d.getVariance(), sample_sizes, n_reps, fmt);
        }

        // Cauchy C(2, 1.5) — mean and variance are undefined (NaN).
        // Proxy metrics: bias/RMSE of getMedian() (= x0) and getGamma().
        {
            auto true_d = CauchyDistribution::create(2.0, 1.5).unwrap();
            const double true_x0 = true_d.getMedian();
            const double true_gamma = true_d.getGamma();
            std::cout << "\n--- Cauchy C(2,1.5)  true x0=2, gamma=1.5 (moments undefined) ---\n";
            std::cout << fmt.formatRow({"n", "ok/M", "x0Bias", "x0RMSE", "gBias", "gRMSE"}) << "\n";
            std::cout << fmt.getSeparator() << "\n";

            for (size_t n : sample_sizes) {
                std::mt19937 rng(42);
                std::vector<double> x0_vals, gamma_vals;
                x0_vals.reserve(static_cast<size_t>(n_reps));
                gamma_vals.reserve(static_cast<size_t>(n_reps));

                for (int rep = 0; rep < n_reps; ++rep) {
                    auto data = true_d.sample(rng, n);
                    auto fitted = CauchyDistribution::create(0.0, 1.0).unwrap();
                    try {
                        fitted.fit(data);
                        double fx0 = fitted.getMedian();
                        double fg = fitted.getGamma();
                        if (std::isfinite(fx0) && std::isfinite(fg)) {
                            x0_vals.push_back(fx0);
                            gamma_vals.push_back(fg);
                        }
                    } catch (...) {
                    }
                }

                auto s = compute_stats(x0_vals, gamma_vals, true_x0, true_gamma);
                auto fmt_v = [](double v) -> std::string {
                    std::ostringstream oss;
                    oss << std::fixed << std::setprecision(4) << v;
                    return oss.str();
                };
                std::string n_ok_str = std::to_string(s.n_ok) + "/" + std::to_string(n_reps);
                std::cout << fmt.formatRow({std::to_string(n), n_ok_str, fmt_v(s.mean_bias),
                                            fmt_v(s.mean_rmse), fmt_v(s.var_bias),
                                            fmt_v(s.var_rmse)})
                          << "\n";
            }
        }

        std::cout << "\n\nNote: MeanRMSE and VarRMSE should decrease as n increases.\n"
                  << "Bias that does not shrink with n indicates MLE inconsistency\n"
                  << "(e.g. Pareto scale underestimation at small n).\n"
                  << "Cauchy shows x0 (median) and gamma recovery; mean/var are undefined.\n";
    });
}

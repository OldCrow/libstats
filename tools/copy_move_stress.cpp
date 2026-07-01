/**
 * @file copy_move_stress.cpp
 * @brief Copy / Move Semantics Concurrent Stress Tool
 *
 * Runs N threads per distribution, each thread continuously creating,
 * copying, and moving distribution objects.  Validates that:
 *
 *   - No deadlocks occur under concurrent copy/move load
 *   - No exceptions are thrown from noexcept move paths
 *   - Computed getMean() values remain finite after copy/move chains
 *   - The factory singleton and cache initialisation machinery are
 *     thread-safe under concurrent construction
 *
 * This complements toctou_validator (which tests concurrent reads + parameter
 * mutations on a *shared* object) by stressing concurrent copy/move of
 * *independently-created* distribution objects across all 19 distributions.
 *
 * @par Usage
 * @code
 * ./build/tools/copy_move_stress [--duration-ms N] [--threads N] [--quick]
 * @endcode
 *
 * Exit code: 0 = all distributions pass, 1 = exception or bad value detected.
 */

#include "tool_utils.h"

#include "libstats/distributions/beta.h"
#include "libstats/distributions/binomial.h"
#include "libstats/distributions/cauchy.h"
#include "libstats/distributions/chi_squared.h"
#include "libstats/distributions/discrete.h"
#include "libstats/distributions/exponential.h"
#include "libstats/distributions/gamma.h"
#include "libstats/distributions/gaussian.h"
#include "libstats/distributions/geometric.h"
#include "libstats/distributions/laplace.h"
#include "libstats/distributions/lognormal.h"
#include "libstats/distributions/negative_binomial.h"
#include "libstats/distributions/pareto.h"
#include "libstats/distributions/poisson.h"
#include "libstats/distributions/rayleigh.h"
#include "libstats/distributions/student_t.h"
#include "libstats/distributions/uniform.h"
#include "libstats/distributions/von_mises.h"
#include "libstats/distributions/weibull.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using namespace stats;

namespace {

constexpr int DEFAULT_DURATION_MS = 300;
constexpr int QUICK_DURATION_MS   = 50;
constexpr int DEFAULT_THREADS     = 8;

// ─── Per-distribution stress result ──────────────────────────────────────────

struct StressResult {
    long long total_ops  = 0;  // complete copy/move cycles
    bool      pass       = true;
    std::string failure_msg;

    long long ops_per_sec(int duration_ms) const {
        return duration_ms > 0 ? (total_ops * 1000LL) / duration_ms : 0;
    }
};

// ─── Core harness ─────────────────────────────────────────────────────────────
//
// Each thread calls make_pair(thread_id) to get two fresh distribution objects,
// then loops: copy-construct, copy-assign, move-construct, getMean() check.
// template parameter Dist is the concrete distribution type.

template<typename Dist>
StressResult run_stress(
    std::function<std::pair<Dist, Dist>(int thread_id)> make_pair,
    int duration_ms,
    int n_threads)
{
    constexpr auto rlx = std::memory_order_relaxed;

    std::atomic<bool>      stop{false};
    std::atomic<long long> total_ops{0};
    std::atomic<bool>      had_error{false};
    std::string            error_msg;

    std::vector<std::thread> threads;
    threads.reserve(static_cast<size_t>(n_threads));

    for (int t = 0; t < n_threads; ++t) {
        threads.emplace_back([&, t] {
            long long local_ops = 0;
            try {
                auto [d1, d2] = make_pair(t);

                while (!stop.load(rlx)) {
                    // Inner loop amortises the stop-flag check
                    for (int i = 0; i < 10 && !stop.load(rlx); ++i) {
                        auto c1 = d1;           // copy-construct
                        auto c2 = d2;           // copy-construct
                        c1 = c2;               // copy-assign
                        c2 = d1;               // copy-assign
                        auto m1 = std::move(c1); // move-construct
                        auto m2 = std::move(c2); // move-construct

                        // Verify CDF is in [0,1] after copy/move chain.
                        // getCumulativeProbability(0.0) is always finite for any valid
                        // distribution state; getMean() is intentionally NaN for
                        // distributions such as Cauchy that have undefined moments.
                        const double cdf1 = m1.getCumulativeProbability(0.0);
                        const double cdf2 = m2.getCumulativeProbability(0.0);
                        if (!std::isfinite(cdf1) || cdf1 < -1e-10 || cdf1 > 1.0 + 1e-10 ||
                            !std::isfinite(cdf2) || cdf2 < -1e-10 || cdf2 > 1.0 + 1e-10) {
                            if (!had_error.exchange(true, rlx)) {
                                std::ostringstream os;
                                os << "getCumulativeProbability(0.0) out of [0,1] after copy/move: "
                                   << cdf1 << ", " << cdf2;
                                error_msg = os.str();
                            }
                        }
                    }
                    ++local_ops;
                }
            } catch (const std::exception& e) {
                if (!had_error.exchange(true, rlx))
                    error_msg = std::string("exception: ") + e.what();
            } catch (...) {
                if (!had_error.exchange(true, rlx))
                    error_msg = "unknown exception";
            }
            total_ops.fetch_add(local_ops, rlx);
        });
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(duration_ms));
    stop.store(true, rlx);
    for (auto& t : threads) t.join();

    StressResult r;
    r.total_ops   = total_ops.load();
    r.pass        = !had_error.load();
    r.failure_msg = error_msg;
    return r;
}

// ─── Per-distribution runners ─────────────────────────────────────────────────

StressResult stress_gaussian(int dur, int nt) {
    return run_stress<GaussianDistribution>([](int t) {
        return std::make_pair(
            GaussianDistribution::create(t * 1.0,      1.0).value,
            GaussianDistribution::create(t * 1.0 + 10, 2.0).value);
    }, dur, nt);
}

StressResult stress_exponential(int dur, int nt) {
    return run_stress<ExponentialDistribution>([](int t) {
        return std::make_pair(
            ExponentialDistribution::create(t + 1.0).value,
            ExponentialDistribution::create(t + 2.0).value);
    }, dur, nt);
}

StressResult stress_uniform(int dur, int nt) {
    return run_stress<UniformDistribution>([](int t) {
        return std::make_pair(
            UniformDistribution::create(t * 10.0, t * 10.0 + 5.0).value,
            UniformDistribution::create(0.0, 1.0).value);
    }, dur, nt);
}

StressResult stress_poisson(int dur, int nt) {
    return run_stress<PoissonDistribution>([](int t) {
        return std::make_pair(
            PoissonDistribution::create(t + 1.0).value,
            PoissonDistribution::create(t + 3.0).value);
    }, dur, nt);
}

StressResult stress_gamma(int dur, int nt) {
    return run_stress<GammaDistribution>([](int t) {
        return std::make_pair(
            GammaDistribution::create(2.0, t + 0.5).value,
            GammaDistribution::create(3.0, t + 1.0).value);
    }, dur, nt);
}

StressResult stress_discrete(int dur, int nt) {
    return run_stress<DiscreteDistribution>([](int t) {
        return std::make_pair(
            DiscreteDistribution::create(1, t + 6).value,
            DiscreteDistribution::create(0, t + 4).value);
    }, dur, nt);
}

StressResult stress_beta(int dur, int nt) {
    return run_stress<BetaDistribution>([](int t) {
        return std::make_pair(
            BetaDistribution::create(1.5 + t * 0.001, 2.5).value,
            BetaDistribution::create(2.5, 1.5 + t * 0.001).value);
    }, dur, nt);
}

StressResult stress_binomial(int dur, int nt) {
    return run_stress<BinomialDistribution>([](int t) {
        return std::make_pair(
            BinomialDistribution::create(t % 20 + 5, 0.3).value,
            BinomialDistribution::create(t % 15 + 8, 0.6).value);
    }, dur, nt);
}

StressResult stress_chi_squared(int dur, int nt) {
    return run_stress<ChiSquaredDistribution>([](int t) {
        return std::make_pair(
            ChiSquaredDistribution::create(2.0 + t * 0.001).value,
            ChiSquaredDistribution::create(5.0 + t * 0.001).value);
    }, dur, nt);
}

StressResult stress_lognormal(int dur, int nt) {
    return run_stress<LogNormalDistribution>([](int t) {
        return std::make_pair(
            LogNormalDistribution::create(0.0, 0.5 + t * 0.001).value,
            LogNormalDistribution::create(0.5, 0.5 + t * 0.001).value);
    }, dur, nt);
}

StressResult stress_negative_binomial(int dur, int nt) {
    return run_stress<NegativeBinomialDistribution>([](int t) {
        return std::make_pair(
            NegativeBinomialDistribution::create(3.0 + t * 0.001, 0.4).value,
            NegativeBinomialDistribution::create(5.0 + t * 0.001, 0.6).value);
    }, dur, nt);
}

StressResult stress_pareto(int dur, int nt) {
    return run_stress<ParetoDistribution>([](int t) {
        return std::make_pair(
            ParetoDistribution::create(1.0, 2.0 + t * 0.001).value,
            ParetoDistribution::create(1.0, 3.0 + t * 0.001).value);
    }, dur, nt);
}

StressResult stress_rayleigh(int dur, int nt) {
    return run_stress<RayleighDistribution>([](int t) {
        return std::make_pair(
            RayleighDistribution::create(1.0 + t * 0.001).value,
            RayleighDistribution::create(2.0 + t * 0.001).value);
    }, dur, nt);
}

StressResult stress_student_t(int dur, int nt) {
    return run_stress<StudentTDistribution>([](int t) {
        return std::make_pair(
            StudentTDistribution::create(3.0 + t * 0.001).value,
            StudentTDistribution::create(5.0 + t * 0.001).value);
    }, dur, nt);
}

StressResult stress_von_mises(int dur, int nt) {
    return run_stress<VonMisesDistribution>([](int t) {
        return std::make_pair(
            VonMisesDistribution::create(0.0, 1.0 + t * 0.001).value,
            VonMisesDistribution::create(1.0, 2.0 + t * 0.001).value);
    }, dur, nt);
}

StressResult stress_weibull(int dur, int nt) {
    return run_stress<WeibullDistribution>([](int t) {
        return std::make_pair(
            WeibullDistribution::create(1.5 + t * 0.001, 1.0).value,
            WeibullDistribution::create(2.5 + t * 0.001, 1.0).value);
    }, dur, nt);
}

StressResult stress_geometric(int dur, int nt) {
    return run_stress<GeometricDistribution>([](int t) {
        return std::make_pair(
            GeometricDistribution::create(0.2 + t * 0.001).value,
            GeometricDistribution::create(0.5 + t * 0.001).value);
    }, dur, nt);
}

StressResult stress_laplace(int dur, int nt) {
    return run_stress<LaplaceDistribution>([](int t) {
        return std::make_pair(
            LaplaceDistribution::create(0.0, 1.0 + t * 0.001).value,
            LaplaceDistribution::create(1.0, 0.5 + t * 0.001).value);
    }, dur, nt);
}

StressResult stress_cauchy(int dur, int nt) {
    return run_stress<CauchyDistribution>([](int t) {
        return std::make_pair(
            CauchyDistribution::create(0.0, 1.0 + t * 0.001).value,
            CauchyDistribution::create(1.0, 0.5 + t * 0.001).value);
    }, dur, nt);
}

// ─── Registry ─────────────────────────────────────────────────────────────────

struct Entry {
    std::string name;
    std::function<StressResult(int dur, int nt)> run;
};

std::vector<Entry> build_registry() {
    return {
        {"Gaussian",         stress_gaussian},
        {"Exponential",      stress_exponential},
        {"Uniform",          stress_uniform},
        {"Poisson",          stress_poisson},
        {"Gamma",            stress_gamma},
        {"Discrete",         stress_discrete},
        {"Beta",             stress_beta},
        {"Binomial",         stress_binomial},
        {"ChiSquared",       stress_chi_squared},
        {"LogNormal",        stress_lognormal},
        {"NegativeBinomial", stress_negative_binomial},
        {"Pareto",           stress_pareto},
        {"Rayleigh",         stress_rayleigh},
        {"StudentT",         stress_student_t},
        {"VonMises",         stress_von_mises},
        {"Weibull",          stress_weibull},
        {"Geometric",        stress_geometric},
        {"Laplace",          stress_laplace},
        {"Cauchy",           stress_cauchy},
    };
}

void print_usage(const char* prog) {
    std::cout <<
        "Usage: " << prog << " [options]\n\n"
        "Options:\n"
        "  --duration-ms N   ms per distribution (default " << DEFAULT_DURATION_MS << ")\n"
        "  --threads N       worker threads per distribution (default " << DEFAULT_THREADS << ")\n"
        "  --quick           50 ms per distribution\n"
        "  --help            print this message\n\n"
        "Exit code: 0 = all pass, 1 = exception or bad value detected\n";
}

}  // namespace

int main(int argc, char** argv) {
    int duration_ms = DEFAULT_DURATION_MS;
    int n_threads   = DEFAULT_THREADS;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--quick") {
            duration_ms = QUICK_DURATION_MS;
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--duration-ms" && i + 1 < argc) {
            duration_ms = std::stoi(argv[++i]);
        } else if (arg == "--threads" && i + 1 < argc) {
            n_threads = std::stoi(argv[++i]);
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    stats::detail::detail::displayToolHeader(
        "Copy / Move Semantics Stress",
        "Concurrent copy/move race-condition correctness test across all 19 distributions");

    std::cout << "Configuration: " << duration_ms << " ms/distribution"
              << ", " << n_threads << " thread(s) per distribution\n\n";

    const auto registry = build_registry();

    // Status is the primary column; copy rate is secondary diagnostic information.
    const int COL_NAME = 20, COL_STATUS = 10, COL_RATE = 20;
    stats::detail::detail::ColumnFormatter fmt({COL_NAME, COL_STATUS, COL_RATE});
    std::cout << fmt.formatRow({"Distribution", "Result", "Copy rate [diagnostic]"}) << "\n";
    std::cout << fmt.getSeparator() << "\n";

    int failures = 0;

    for (const auto& entry : registry) {
        std::cout << std::left << std::setw(COL_NAME) << entry.name << std::flush;

        const StressResult r = entry.run(duration_ms, n_threads);
        const long long ops  = r.ops_per_sec(duration_ms);

        // Status first.
        // ✓/✗ are 3-byte UTF-8 but 1-char display width: std::setw counts bytes,
        // not display chars, so it under-pads by 2. Use literal spacing instead.
        if (r.pass) {
            std::cout << "PASS ✓    ";  // 4 trailing spaces → 10 display chars
        } else {
            std::cout << "FAIL ✗    ";
            ++failures;
        }

        // Copy rate — secondary, labelled to clarify it is not a benchmark target
        std::string rate_str;
        {
            std::ostringstream os;
            std::string s = std::to_string(ops);
            int ins = static_cast<int>(s.size()) % 3;
            for (int i = 0; i < static_cast<int>(s.size()); ++i) {
                if (i > 0 && (i - ins) % 3 == 0) os << ',';
                os << s[static_cast<size_t>(i)];
            }
            os << " cycles/s";
            rate_str = os.str();
        }
        std::cout << rate_str << "\n";

        if (!r.pass)
            std::cout << "  [ERROR] " << r.failure_msg << "\n";
    }

    std::cout << fmt.getSeparator() << "\n";

    if (failures == 0) {
        std::cout << "\n✓ All " << registry.size() << " distributions PASS"
                  << " — no deadlocks, no exceptions, CDF always in [0,1]\n\n";
    } else {
        std::cout << "\n✗ " << failures << " of " << registry.size()
                  << " distributions FAIL\n\n";
    }

    std::cout << "Notes:\n"
              << "  RESULT is the primary output: PASS = no race conditions detected\n"
              << "  under concurrent copy/move load (no exceptions, no bad CDF values,\n"
              << "  no deadlocks). Exit code reflects RESULT only.\n"
              << "\n"
              << "  Copy rate [diagnostic] is a secondary signal showing how many\n"
              << "  copy/move cycles each thread completed per second. Low rates\n"
              << "  indicate expensive copy constructors (e.g. VonMises copies its\n"
              << "  2049-point CDF grid), not race conditions. This column is useful\n"
              << "  for identifying copy/move optimisation candidates post-v2.0.0.\n\n";

    return (failures > 0) ? 1 : 0;
}

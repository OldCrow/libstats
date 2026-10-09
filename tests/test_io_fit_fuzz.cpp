/**
 * @file test_io_fit_fuzz.cpp
 * @brief Stream round trip and fit-on-degenerate-data contracts for all 27 distributions.
 *
 *   stream   operator<< then operator>> at awkward parameter values (below 5e-7, not
 *            representable in decimal, 17 significant digits, infinite truncation bounds):
 *            - reads back without failbit, and writing the result reproduces the text;
 *            - restores the parameters grossly: mean, variance and support bounds within 10% of
 *              their own scale, with no collapse to 0 and no finite/non-finite change. Any print
 *              precision passes; a parameter printed as 0.000000, or read from the wrong field,
 *              does not. (Instances keep 1 - p above 1e-6 so 6 significant digits cannot change
 *              the family's shape.);
 *            - restores them exactly (every moment and density in the fingerprint bit for bit):
 *              every class writes max_digits10 since #208.
 *   garbage  operator>> on truncated and mutated text never leaves invalid parameters, and when
 *            it sets failbit it leaves the object unchanged.
 *   fit      on degenerate data (empty, one point, all equal, NaN, +-inf, negative, huge, tiny,
 *            zeros, out of support, DBL_MAX, denormals) either throws or leaves valid parameters
 *            with non-NaN moments (where the family defines them) and a CDF in [0, 1]; non-finite
 *            data always throws; the fitted object round-trips through the stream as above.
 *
 * Data and probe points are the instance's own quantiles at fixed p, so the run is identical on
 * every standard library. Fitted objects are probed without quantile calls (a fit can land on
 * parameters where the quantile solver is slow, which is not what this test measures).
 */

#define LIBSTATS_FULL_INTERFACE
#include "libstats/libstats.h"

#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <gtest/gtest.h>
#include <iomanip>
#include <limits>
#include <map>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

using P = std::array<double, 4>;
constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

std::string fmt(double v) {
    std::ostringstream o;
    o << std::setprecision(17) << v;
    return o.str();
}
bool sameBits(double a, double b) {
    return (std::isnan(a) && std::isnan(b)) || std::memcmp(&a, &b, sizeof(double)) == 0;
}

// Families whose mean or variance is undefined for part of the parameter space (NaN or inf by
// design there).
bool mayHaveUndefinedMoments(const std::string& family) {
    return family == "cauchy" || family == "student_t" || family == "fisher_f" ||
           family == "inverse_gamma" || family == "pareto";
}

template <typename D>
double safe(const std::function<double()>& f) {
    try {
        return f();
    } catch (...) {
        return -777.0;
    }
}

// Parameter proxies: mean, variance, support bounds. Together they pin both parameters of every
// two-parameter family (and the bounds pin the truncation window).
template <typename D>
std::vector<double> proxies(const D& d) {
    return {safe<D>([&] { return d.getMean(); }), safe<D>([&] { return d.getVariance(); }),
            safe<D>([&] { return d.getSupportLowerBound(); }),
            safe<D>([&] { return d.getSupportUpperBound(); })};
}

// Everything observable the parameters determine, at fixed probe points (no quantile calls).
template <typename D>
std::vector<double> fingerprint(const D& d, const std::vector<double>& probes) {
    std::vector<double> v = proxies(d);
    v.push_back(safe<D>([&] { return d.getSkewness(); }));
    v.push_back(safe<D>([&] { return d.getKurtosis(); }));
    for (double x : probes) {
        v.push_back(safe<D>([&] { return d.getProbability(x); }));
        v.push_back(safe<D>([&] { return d.getLogProbability(x); }));
        v.push_back(safe<D>([&] { return d.getCumulativeProbability(x); }));
    }
    return v;
}

struct Failures {
    std::map<std::string, std::vector<std::string>> byCheck;  // "check | instance"
    std::map<std::string, int> known;
    void add(const std::string& check, const std::string& inst, const std::string& detail) {
        auto& v = byCheck[check + " | " + inst];
        if (v.size() < 3)
            v.push_back(detail);
        else if (v.size() == 3)
            v.push_back("...");
    }
    void addKnown(const std::string& tag, const std::string& check, const std::string& inst) {
        known["#" + tag + " " + check + " | " + inst]++;
    }
    void finish() {
        for (const auto& [k, v] : byCheck) {
            std::string s = "[" + k + "]";
            for (const auto& l : v)
                s += "\n    " + l;
            ADD_FAILURE() << s;
        }
        std::string list;
        for (const auto& [k, n] : known) {
            std::printf("[ known    ] %s (%d)\n", k.c_str(), n);
            list += "\n  " + k + " (" + std::to_string(n) + ")";
        }
        if (!known.empty() && !::testing::Test::HasFailure())
            GTEST_SKIP() << "only known open defects failed:" << list;
    }
};

// A count family with 1 - p below 1e-4 changes shape when p is printed to 6 significant digits (a
// fitted p = 0.9999955 prints as 0.999995: 1 - p moves 10%). Its gross tier is skipped (the exact
// tier still runs); p is recovered from the mean and variance. A small p keeps its relative
// precision in any significant-digit format, so it stays in the gross tier.
bool printSensitive(const std::string& family, const std::vector<double>& px) {
    const double mean = px[0], var = px[1];
    double p = 0.5;
    if (family == "negative_binomial" || family == "geometric")
        p = mean / var;
    else if (family == "binomial")
        p = 1.0 - var / mean;
    else if (family == "bernoulli")
        p = mean;
    else
        return false;
    return !(1.0 - p >= 1e-4);
}

// Writes d, reads it back, and scores the three tiers.
template <typename D>
void streamRoundTrip(const D& d, const std::vector<double>& probes, const std::string& family,
                     const std::string& check, const std::string& inst, const std::string& prefix,
                     Failures& fl) {
    std::ostringstream os;
    os << d;
    const std::string text = os.str();
    D e;
    std::istringstream is(text);
    is >> e;
    if (is.fail()) {
        fl.add(check, inst, prefix + "operator>> failed on \"" + text + "\"");
        return;
    }
    if (!e.validateCurrentParameters().isOk()) {
        fl.add(check, inst, prefix + "read back invalid parameters from \"" + text + "\"");
        return;
    }
    std::ostringstream os2;
    os2 << e;
    if (os2.str() != text) {
        fl.add(check, inst, prefix + "\"" + text + "\" read back as \"" + os2.str() + "\"");
        return;
    }
    const auto a = proxies(d), b = proxies(e);
    const bool gross = !printSensitive(family, a);
    const double scale = std::isfinite(a[1]) && a[1] > 0 ? std::sqrt(a[1]) : 0.0;
    for (std::size_t i = 0; gross && i < a.size(); ++i) {
        if (sameBits(a[i], b[i]))
            continue;
        const double tol = 0.1 * std::max({std::fabs(a[i]), i == 1 ? 0.0 : scale, 1e-300});
        const bool collapsed =
            (a[i] != 0.0 && b[i] == 0.0) || std::isfinite(a[i]) != std::isfinite(b[i]);
        if (collapsed || !(std::fabs(a[i] - b[i]) <= tol)) {
            fl.add(check, inst,
                   prefix + "proxy[" + std::to_string(i) + "] " + fmt(a[i]) + " -> " + fmt(b[i]) +
                       " via \"" + text + "\"");
            return;
        }
    }
    const auto fa = fingerprint(d, probes), fb = fingerprint(e, probes);
    for (std::size_t i = 0; i < fa.size(); ++i)
        if (!sameBits(fa[i], fb[i])) {
            // Every class writes max_digits10 (#208), so the round trip is exact.
            fl.add(check + " (exact)", inst,
                   prefix + "probe " + std::to_string(i) + " differs after \"" + text + "\"");
            return;
        }
}

template <typename D>
void streamChecks(const D& d, const std::vector<double>& probes, const std::string& family,
                  const std::string& inst, Failures& fl, std::mt19937_64& g) {
    streamRoundTrip(d, probes, family, "stream round trip", inst, "", fl);

    std::ostringstream os;
    os << d;
    const std::string text = os.str();
    std::vector<std::string> inputs = {"", "garbage", text + " trailing"};
    for (std::size_t i = 0; i < text.size(); i += 3)
        inputs.push_back(text.substr(0, i));
    static const char alphabet[] = "0123456789.-+eE nainfNaN,=()[]:;";
    for (int k = 0; k < 60; ++k) {
        std::string t = text;
        const int n = 1 + static_cast<int>(g() % 3);
        for (int m = 0; m < n; ++m)
            t[g() % t.size()] = alphabet[g() % (sizeof(alphabet) - 1)];
        inputs.push_back(t);
    }
    const auto before = fingerprint(d, probes);
    for (const auto& in : inputs) {
        D e = d;
        std::istringstream is(in);
        bool threw = false;
        try {
            is >> e;
        } catch (const std::exception&) {
            threw = true;
        }
        if (!e.validateCurrentParameters().isOk())
            fl.add("garbage left invalid parameters", inst, "input \"" + in + "\"");
        if (is.fail() && !threw) {
            const auto after = fingerprint(e, probes);
            for (std::size_t i = 0; i < after.size(); ++i)
                if (!sameBits(after[i], before[i])) {
                    fl.add("garbage: failbit but object changed", inst, "input \"" + in + "\"");
                    break;
                }
        }
    }
}

template <typename D>
void fitChecks(const D& d, const std::vector<double>& sample, const std::string& family,
               const std::string& inst, Failures& fl) {
    const double lo = d.getSupportLowerBound(), hi = d.getSupportUpperBound();
    std::vector<std::pair<std::string, std::vector<double>>> sets = {
        {"empty", {}},
        {"one", {sample[17]}},
        {"two-equal", {sample[21], sample[21]}},
        {"all-equal", std::vector<double>(40, sample[33])},
        {"zeros", std::vector<double>(30, 0.0)},
        {"ones", std::vector<double>(30, 1.0)},
        {"out-of-support-high", std::vector<double>(30, std::isinf(hi) ? 1e10 : hi + 1)},
        {"out-of-support-low", std::vector<double>(30, std::isinf(lo) ? -1e10 : lo - 1)},
        {"halfint", {0.5, 1.5, 2.5}},
        {"dblmax", std::vector<double>(10, DBL_MAX)},
        {"denorm", std::vector<double>(10, std::numeric_limits<double>::denorm_min())},
    };
    auto scaled = [&](double s) {
        auto v = sample;
        for (auto& x : v)
            x *= s;
        return v;
    };
    sets.emplace_back("huge", scaled(1e300));
    sets.emplace_back("tiny", scaled(1e-300));
    {
        auto v = sample;
        for (std::size_t i = 0; i < v.size(); ++i)
            v[i] = -std::fabs(v[i]) - 1.0 - static_cast<double>(i);
        sets.emplace_back("negative", v);
    }
    // Non-finite data must be rejected, not dropped (AR D4).
    std::vector<std::string> mustThrow;
    for (double bad : {kNaN, kInf, -kInf}) {
        auto v = sample;
        v[7] = bad;
        const std::string name = std::string("with-") + (std::isnan(bad) ? "NaN"
                                                         : bad > 0       ? "inf"
                                                                         : "-inf");
        sets.emplace_back(name, v);
        mustThrow.push_back(name);
    }

    for (const auto& [name, data] : sets) {
        D e = d;
        bool threw = false;
        try {
            e.fit(data);
        } catch (const std::exception&) {
            threw = true;
        }
        const bool needThrow =
            std::find(mustThrow.begin(), mustThrow.end(), name) != mustThrow.end();
        if (needThrow && !threw) {
            std::ostringstream os;
            os << e;
            fl.add("fit accepted non-finite data", inst, name + " -> " + os.str());
        }
        if (threw)
            continue;
        std::ostringstream os;
        os << e;
        if (!e.validateCurrentParameters().isOk()) {
            fl.add("fit left invalid parameters", inst, name + " -> " + os.str());
            continue;
        }
        double mean = kNaN, var = kNaN, Fx = kNaN;
        try {
            mean = e.getMean();
            var = e.getVariance();
            Fx = e.getCumulativeProbability(data.empty() ? 0.0 : data[0]);
        } catch (const std::exception& ex) {
            fl.add("fitted object throws", inst, name + " -> " + os.str() + ": " + ex.what());
            continue;
        }
        if (!(Fx >= 0.0 && Fx <= 1.0))
            fl.add("fitted CDF outside [0, 1]", inst, name + " -> " + os.str() + " F=" + fmt(Fx));
        if (!mayHaveUndefinedMoments(family) && (std::isnan(mean) || std::isnan(var)))
            fl.add("fitted moments NaN", inst,
                   name + " -> " + os.str() + " mean=" + fmt(mean) + " var=" + fmt(var));
        streamRoundTrip(e, sample, family, "fitted object stream round trip", inst, name + ": ",
                        fl);
    }
}

template <typename D, typename Make>
void runFamily(const std::string& family, Make make, const std::vector<P>& instances) {
    Failures fl;
    std::mt19937_64 g(0x10f17ULL + family.size());
    for (const auto& par : instances) {
        std::ostringstream tag;
        tag << family << "(" << std::setprecision(17) << par[0] << "," << par[1];
        if (family == "truncated_normal")
            tag << "," << par[2] << "," << par[3];
        tag << ")";
        std::optional<D> d;
        try {
            d.emplace(make(par));
        } catch (const std::exception& e) {
            ADD_FAILURE() << tag.str() << ": constructor threw " << e.what();
            continue;
        }
        // Data and probe points: the instance's own quantiles at fixed p.
        std::vector<double> sample;
        for (int i = 0; i < 50; ++i)
            sample.push_back(d->getQuantile((i + 0.5) / 50.0));
        streamChecks(*d, sample, family, tag.str(), fl, g);
        fitChecks(*d, sample, family, tag.str(), fl);
    }
    fl.finish();
}

// Awkward values: 1/3 and 0.1 (no finite decimal), pi*1e-7 (below the 6-decimal resolution),
// 1 + 2^-52 (17 significant digits), 98765.4321012345 (large with a long fraction).
const double kThird = 1.0 / 3.0;
const double kSmall = 3.14159265358979e-7;
const double kOnePlus = 1.0 + std::numeric_limits<double>::epsilon();
const double kLong = 98765.4321012345;

using namespace stats;

#define LOC_SCALE_SET                                                                              \
    {{kThird, 0.1}, {-kLong, kSmall}, {0.1, kOnePlus}, {kSmall, kLong}, {0.0, 1.0}}
#define SCALE_SET                                                                                  \
    {                                                                                              \
        {kThird}, {0.1}, {kSmall}, {kOnePlus}, {                                                   \
            kLong                                                                                  \
        }                                                                                          \
    }
#define SHAPE_SCALE_SET                                                                            \
    {{kThird, 0.1}, {kLong, kSmall}, {0.1, kOnePlus}, {kSmall, kLong}, {2.0, 1.0}}

TEST(IoFitFuzz, Gaussian) {
    runFamily<GaussianDistribution>(
        "gaussian", [](const P& p) { return GaussianDistribution(p[0], p[1]); }, LOC_SCALE_SET);
}
TEST(IoFitFuzz, LogNormal) {
    runFamily<LogNormalDistribution>(
        "lognormal", [](const P& p) { return LogNormalDistribution(p[0], p[1]); },
        {{kThird, 0.1}, {-3.5, kSmall}, {0.1, kOnePlus}, {kSmall, 2.5}});
}
TEST(IoFitFuzz, Exponential) {
    runFamily<ExponentialDistribution>(
        "exponential", [](const P& p) { return ExponentialDistribution(p[0]); }, SCALE_SET);
}
TEST(IoFitFuzz, Uniform) {
    runFamily<UniformDistribution>(
        "uniform", [](const P& p) { return UniformDistribution(p[0], p[1]); },
        {{kThird, kOnePlus}, {-kLong, kSmall}, {0.0, kSmall}, {0.1, kLong}});
}
TEST(IoFitFuzz, Gamma) {
    runFamily<GammaDistribution>(
        "gamma", [](const P& p) { return GammaDistribution(p[0], p[1]); }, SHAPE_SCALE_SET);
}
TEST(IoFitFuzz, StudentT) {
    runFamily<StudentTDistribution>("student_t",
                                    [](const P& p) { return StudentTDistribution(p[0]); },
                                    {{kThird}, {2 + kThird}, {kLong}, {5.0}});
}
TEST(IoFitFuzz, Cauchy) {
    runFamily<CauchyDistribution>(
        "cauchy", [](const P& p) { return CauchyDistribution(p[0], p[1]); }, LOC_SCALE_SET);
}
TEST(IoFitFuzz, VonMises) {
    runFamily<VonMisesDistribution>(
        "von_mises", [](const P& p) { return VonMisesDistribution(p[0], p[1]); },
        {{kThird, 0.1}, {-2.5, kSmall}, {0.1, kOnePlus}, {kSmall, 50.5}});
}
TEST(IoFitFuzz, Binomial) {
    runFamily<BinomialDistribution>(
        "binomial", [](const P& p) { return BinomialDistribution(static_cast<int>(p[0]), p[1]); },
        {{20, kThird}, {7, 0.1}, {1000, kSmall}, {3, 0.9 + kSmall}});
}
TEST(IoFitFuzz, NegativeBinomial) {
    runFamily<NegativeBinomialDistribution>(
        "negative_binomial", [](const P& p) { return NegativeBinomialDistribution(p[0], p[1]); },
        {{kThird, 0.1}, {kLong, kThird}, {5.0, kSmall * 1e3}, {kSmall, 0.5}});
}
TEST(IoFitFuzz, Geometric) {
    runFamily<GeometricDistribution>("geometric",
                                     [](const P& p) { return GeometricDistribution(p[0]); },
                                     {{kThird}, {0.1}, {kSmall}, {0.9 + kSmall}});
}
TEST(IoFitFuzz, Beta) {
    runFamily<BetaDistribution>("beta", [](const P& p) { return BetaDistribution(p[0], p[1]); },
                                {{kThird, 0.1}, {2.5, kOnePlus}, {0.1, kLong / 1e3}});
}
TEST(IoFitFuzz, ChiSquared) {
    runFamily<ChiSquaredDistribution>("chi_squared",
                                      [](const P& p) { return ChiSquaredDistribution(p[0]); },
                                      {{kThird}, {0.1}, {kOnePlus}, {kLong}});
}
TEST(IoFitFuzz, Laplace) {
    runFamily<LaplaceDistribution>(
        "laplace", [](const P& p) { return LaplaceDistribution(p[0], p[1]); }, LOC_SCALE_SET);
}
TEST(IoFitFuzz, Pareto) {
    runFamily<ParetoDistribution>(
        "pareto", [](const P& p) { return ParetoDistribution(p[0], p[1]); },
        {{kThird, 0.1}, {kSmall, 2.5}, {kOnePlus, kThird}, {kLong, 1 + kThird}});
}
TEST(IoFitFuzz, Rayleigh) {
    runFamily<RayleighDistribution>(
        "rayleigh", [](const P& p) { return RayleighDistribution(p[0]); }, SCALE_SET);
}
TEST(IoFitFuzz, Weibull) {
    runFamily<WeibullDistribution>(
        "weibull", [](const P& p) { return WeibullDistribution(p[0], p[1]); },
        {{kThird, 0.1}, {2.5, kSmall}, {0.1 * 10, kOnePlus}, {kLong / 1e4, kLong}});
}
TEST(IoFitFuzz, Logistic) {
    runFamily<LogisticDistribution>(
        "logistic", [](const P& p) { return LogisticDistribution(p[0], p[1]); }, LOC_SCALE_SET);
}
TEST(IoFitFuzz, Gumbel) {
    runFamily<GumbelDistribution>(
        "gumbel", [](const P& p) { return GumbelDistribution(p[0], p[1]); }, LOC_SCALE_SET);
}
TEST(IoFitFuzz, Erlang) {
    runFamily<ErlangDistribution>(
        "erlang", [](const P& p) { return ErlangDistribution(static_cast<int>(p[0]), p[1]); },
        {{3, kThird}, {1, kSmall}, {20, kOnePlus}, {7, kLong}});
}
TEST(IoFitFuzz, FisherF) {
    runFamily<FDistribution>("fisher_f", [](const P& p) { return FDistribution(p[0], p[1]); },
                             {{kThird, 0.1 * 50}, {5.0, kOnePlus}, {kLong / 1e3, 12.5}});
}
TEST(IoFitFuzz, InverseGamma) {
    runFamily<InverseGammaDistribution>(
        "inverse_gamma", [](const P& p) { return InverseGammaDistribution(p[0], p[1]); },
        SHAPE_SCALE_SET);
}
TEST(IoFitFuzz, HalfNormal) {
    runFamily<HalfNormalDistribution>(
        "half_normal", [](const P& p) { return HalfNormalDistribution(p[0]); }, SCALE_SET);
}
TEST(IoFitFuzz, Bernoulli) {
    runFamily<BernoulliDistribution>("bernoulli",
                                     [](const P& p) { return BernoulliDistribution(p[0]); },
                                     {{kThird}, {0.1}, {kSmall}, {0.9 + kSmall}});
}
TEST(IoFitFuzz, Poisson) {
    runFamily<PoissonDistribution>("poisson", [](const P& p) { return PoissonDistribution(p[0]); },
                                   {{kThird}, {kSmall}, {kOnePlus}, {kLong}});
}
TEST(IoFitFuzz, Discrete) {
    runFamily<DiscreteDistribution>("discrete",
                                    [](const P& p) {
                                        return DiscreteDistribution(static_cast<int>(p[0]),
                                                                    static_cast<int>(p[1]));
                                    },
                                    {{0, 9}, {-5, 5}, {-1000000, 1000000}, {7, 7}});
}
TEST(IoFitFuzz, TruncatedNormal) {
    // operator>> read the lower bound from "sigma=" (D12).
    runFamily<TruncatedNormalDistribution>(
        "truncated_normal",
        [](const P& p) { return TruncatedNormalDistribution(p[0], p[1], p[2], p[3]); },
        {{0, 1, -2, 2},
         {0, 1, 0, kInf},
         {kThird, 0.1, -kInf, 0.5},
         {0, 2, 10, 12},
         {1000, 100, -kInf, kInf},
         {-kSmall, kOnePlus, -kThird, kLong}});
}

}  // namespace

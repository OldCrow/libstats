#include "seam.h"

static std::vector<double> shapeGrid() {
    std::vector<double> v;
    for (double a : {1e-3, 0.004, 0.01, 0.05,  0.1,   0.3, 0.49, 0.5,    0.51,   0.7,  0.9,
                     1.0,  1.5,   2.0,  3.3,   5.0,   7.7, 10.0, 19.0,   19.999, 20.0, 20.001,
                     25.0, 50.0,  99.5, 200.0, 777.0, 1e3, 3e3,  9999.0, 9999.99})
        v.push_back(a);
    return v;
}

template <class D>
void cdfSeam1(const std::string& lab, const std::string& ctx, const D& d, double x0, int N = 48) {
    walk(
        lab + " CDF", ctx, [&](double x) { return d.getCumulativeProbability(x); }, x0, +1, true,
        [&](double x) { return d.getProbability(x); }, N);
    walk(
        lab + " SF", ctx, [&](double x) { return d.getSurvival(x); }, x0, -1, true,
        [&](double x) { return -d.getProbability(x); }, N);
    walk(
        lab + " logPDF", ctx, [&](double x) { return d.getLogProbability(x); }, x0, 0, false, {},
        N);
    fsCheck(
        lab + " F+S", ctx, [&](double x) { return d.getCumulativeProbability(x); },
        [&](double x) { return d.getSurvival(x); }, x0);
}

template <class D>
void cdfSeam(const std::string& lab, const std::string& ctx, const D& d, double x0, int N = 48) {
    cdfSeam1(lab, ctx, d, x0, N);
    cdfSeam1(lab + " [ctrl]", ctx, d, x0 * (1 + 3e-3), N);
    cdfSeam1(lab + " [ctrl]", ctx, d, x0 * (1 - 3e-3), N);
}
template <class D>
void quantSeam1(const std::string& lab, const std::string& ctx, const D& d, double p0, int N = 48) {
    walk(
        lab, ctx, [&](double p) { return d.getQuantile(p); }, p0, +1, false,
        [&](double p) { return 1.0 / d.getProbability(d.getQuantile(p)); }, N, 36, 0x1p-1074, 1.0);
}

static double temmeEdgeMu(double target, bool upper) {
    // -2(log1p(mu) - mu) = target (eta^2), mu>0 for upper
    auto g = [](double mu) { return -2.0 * (std::log1p(mu) - mu); };
    double lo = upper ? 0.0 : -0.99, hi = upper ? 1.0 : 0.0;
    for (int i = 0; i < 200; ++i) {
        double m = 0.5 * (lo + hi);
        bool over = g(m) > target;
        if (upper) {
            if (over)
                hi = m;
            else
                lo = m;
        } else {
            if (over)
                lo = m;
            else
                hi = m;
        }
    }
    return 0.5 * (lo + hi);
}

template <class D>
void quantSeam(const std::string& lab, const std::string& ctx, const D& d, double p0, int N = 48) {
    quantSeam1(lab, ctx, d, p0, N);
    quantSeam1(lab + " [ctrl]", ctx, d, p0 * (1 + 3e-3), N);
    quantSeam1(lab + " [ctrl]", ctx, d, p0 * (1 - 3e-3), N);
}
int main() {
    // G1: series/CF switch x = a + 1 (also Q small-shape path below 0.5), rate 1 and rate 3
    for (double a : shapeGrid()) {
        for (double rate : {1.0, 3.0}) {
            auto r = GammaDistribution::create(a, rate);
            if (r.isError())
                continue;
            auto d = r.unwrap();
            cdfSeam("Gamma x=(a+1)/rate", fmt("Gamma(%.17g,%g)", a, rate), d, (a + 1) / rate);
        }
    }
    // G2: Temme zone eta = +-0.6 for a >= 1e4, and the a = 1e4 zone entry
    for (double a : {1e4, 1.0000001e4, 3e4, 1e5, 1e6, 1e8, 1e10, 1e12, 1e15}) {
        for (bool up : {true, false}) {
            const double mu = temmeEdgeMu(0.36, up);
            const double x0 = a * (1 + mu);
            auto d = GammaDistribution::create(a, 1.0).unwrap();
            cdfSeam("Gamma Temme |eta|=0.6", fmt("Gamma(%.17g,1)", a), d, x0, 64);
        }
    }
    // G3: parameter seams in a: 0.5 (small-shape Q), 20 (Stirling prefactor), 1e4 (Temme), 1e16 (WH
    // quantile)
    for (double a0 : {0.5, 20.0, 1e4}) {
        for (double xr : {0.2, 0.9, 1.0, 1.05, 1.3, 2.0}) {
            for (double rate : {1.0, 3.0}) {
                const double x = (a0 * xr + (a0 < 1 ? 0.3 : 0.0)) / rate;
                auto cdf = [&](double a) {
                    return GammaDistribution::create(a, rate).unwrap().getCumulativeProbability(x);
                };
                auto sf = [&](double a) {
                    return GammaDistribution::create(a, rate).unwrap().getSurvival(x);
                };
                auto lp = [&](double a) {
                    return GammaDistribution::create(a, rate).unwrap().getLogProbability(x);
                };
                std::string ctx = fmt("Gamma(a~%g,%g) x=%.17g", a0, rate, x);
                walk(fmt("Gamma param a=%g CDF(a)", a0), ctx, cdf, a0, -1, true, {}, 48, 30);
                walk(fmt("Gamma param a=%g SF(a)", a0), ctx, sf, a0, +1, true, {}, 48, 30);
                walk(fmt("Gamma param a=%g logPDF(a)", a0), ctx, lp, a0, 0, false, {}, 48, 30);
            }
        }
    }
    for (double p : {1e-300, 1e-10, 0.1, 0.5, 0.9, 1 - 1e-12}) {
        auto q = [&](double a) {
            return GammaDistribution::create(a, 1.0).unwrap().getQuantile(p);
        };
        walk("Gamma param a=1e16 Q(a)", fmt("p=%g", p), q, 1e16, +1, false, {}, 48, 30);
        walk("Gamma param a=20 Q(a)", fmt("p=%g", p), q, 20.0, +1, false, {}, 48, 30);
        walk("Gamma param a=1e4 Q(a)", fmt("p=%g", p), q, 1e4, +1, false, {}, 48, 30);
    }
    // G4: y = beta*x = DBL_MIN switch (log-x form below)
    for (double a : {0.005, 0.1, 0.5, 1.0, 2.0}) {
        for (double rate : {0.5, 3.0, 1e-10}) {
            auto d = GammaDistribution::create(a, rate).unwrap();
            const double x0 = std::numeric_limits<double>::min() / rate;
            cdfSeam("Gamma y=DBL_MIN", fmt("Gamma(%g,%g)", a, rate), d, x0);
        }
    }
    // G5: log1pmx series limit |t| = 1/2 in the Stirling prefactor (a >= 20): x = 1.5a, 0.5a
    for (double a : {20.0, 37.5, 100.0, 1e3, 9999.0, 1e5}) {
        for (double f : {0.5, 1.5}) {
            for (double rate : {1.0, 3.0}) {
                auto d = GammaDistribution::create(a, rate).unwrap();
                cdfSeam("Gamma |t|=1/2", fmt("Gamma(%g,%g)", a, rate), d, a * f / rate);
                walk(
                    "Gamma |t|=1/2 PDF", fmt("Gamma(%g,%g)", a, rate),
                    [&](double x) { return d.getProbability(x); }, a * f / rate, 0, false);
            }
        }
    }
    // G6: quantile across p = 1/2 (lower/upper solve) and the t_bound = -700 shortcut
    for (double a : shapeGrid()) {
        auto d = GammaDistribution::create(a, 1.0).unwrap();
        quantSeam("Gamma Q p=1/2", fmt("Gamma(%.17g,1)", a), d, 0.5);
        const double lp = -700.0 * a - std::lgamma(a + 1);
        if (lp > -744.0)
            quantSeam("Gamma Q t_bound=-700", fmt("Gamma(%.17g,1)", a), d, std::exp(lp));
    }
    // ChiSquared, Erlang, InverseGamma through Gamma
    for (double k : {0.01, 1.0, 3.0, 39.0, 40.0, 41.0, 1000.0, 2e4}) {
        auto d = ChiSquaredDistribution::create(k).unwrap();
        cdfSeam("ChiSq x=2(a+1)", fmt("ChiSq(%g)", k), d, 2 * (k / 2 + 1));
        quantSeam("ChiSq Q p=1/2", fmt("ChiSq(%g)", k), d, 0.5);
    }
    for (int k : {1, 2, 5, 19, 20, 21, 100}) {
        auto d = ErlangDistribution::create(k, 3.0).unwrap();
        cdfSeam("Erlang x=(k+1)/rate", fmt("Erlang(%g,3)", k), d, (k + 1) / 3.0);
        quantSeam("Erlang Q p=1/2", fmt("Erlang(%g,3)", k), d, 0.5);
    }
    for (double a : {0.01, 0.3, 0.5, 1.0, 3.0, 20.0, 50.0, 1e4}) {
        for (double b : {1.0, 3.0}) {
            auto d = InverseGammaDistribution::create(a, b).unwrap();
            cdfSeam("InvGamma b/x=a+1", fmt("InvGamma(%g,%g)", a, b), d, b / (a + 1));
            quantSeam("InvGamma Q p=1/2", fmt("InvGamma(%g,%g)", a, b), d, 0.5);
        }
    }
    report();
}

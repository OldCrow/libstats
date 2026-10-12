#include "seam.h"

static BetaDistribution mk(double a, double b) {
    return BetaDistribution::create(a, b).unwrap();
}

static void cdf1(const std::string& lab, const std::string& ctx, const BetaDistribution& d,
                 double x0) {
    walk(
        lab + " CDF", ctx, [&](double x) { return d.getCumulativeProbability(x); }, x0, +1, true,
        [&](double x) { return d.getProbability(x); }, 48, 36, 0x1p-1074, 1.0);
    walk(
        lab + " logPDF", ctx, [&](double x) { return d.getLogProbability(x); }, x0, 0, false, {},
        48, 36, 0x1p-1074, std::nextafter(1.0, 0.0));
}
static void cdfSeam(const std::string& lab, const std::string& ctx, const BetaDistribution& d,
                    double x0) {
    cdf1(lab, ctx, d, x0);
    cdf1(lab + " [ctrl]", ctx, d, x0 * (1 + 3e-3) < 1 ? x0 * (1 + 3e-3) : x0);
    cdf1(lab + " [ctrl]", ctx, d, x0 * (1 - 3e-3));
}
static void q1(const std::string& lab, const std::string& ctx, const BetaDistribution& d,
               double p0) {
    walk(
        lab, ctx, [&](double p) { return d.getQuantile(p); }, p0, +1, false,
        [&](double p) { return 1.0 / d.getProbability(d.getQuantile(p)); }, 48, 36, 0x1p-1074, 1.0);
}
static void qSeam(const std::string& lab, const std::string& ctx, const BetaDistribution& d,
                  double p0) {
    q1(lab, ctx, d, p0);
    q1(lab + " [ctrl]", ctx, d, p0 * (1 + 3e-3));
    q1(lab + " [ctrl]", ctx, d, p0 * (1 - 3e-3));
}

int main() {
    const std::vector<double> shapes = {1e-300, 1e-30, 1e-6, 9.99e-4, 1e-3, 1.001e-3, 0.01,  0.1,
                                        0.5,    0.999, 1.0,  1.001,   2.0,  5.0,      19.99, 20.0,
                                        20.01,  50.0,  1e3,  1e6,     3e8,  1e12};
    for (double a : shapes)
        for (double b : shapes) {
            auto r = BetaDistribution::create(a, b);
            if (r.isError())
                continue;
            auto d = r.unwrap();
            const std::string ctx = fmt("Beta(%.17g,%.17g)", a, b);
            // B1 branch point (from the smaller side, as the library decides it)
            const double xb = (a + 1) / (a + b + 2);
            const double omxb = (b + 1) / (a + b + 2);
            const double x0 = xb <= 0.5 ? xb : 1.0 - omxb;
            if (x0 > 0 && x0 < 1)
                cdfSeam("Beta branch x_b", ctx, d, x0);
            // B2 the CDF = 1/2 point (tail-integral <= 1/2 rule for b < 1, symmetric tie for a ==
            // b)
            const double xm = bisectD(0x1p-1074, std::nextafter(1.0, 0.0), [&](double x) {
                return d.getCumulativeProbability(x) >= 0.5;
            });
            if (xm > 0 && xm < 1)
                cdfSeam("Beta F=1/2 point", ctx, d, xm);
            // B6 |u| = 1/2 in the Stirling prefactor
            if (a >= 20 && b >= 20) {
                const double xs = a / (a + b);
                for (double f : {0.5, 1.5})
                    if (xs * f < 1)
                        cdfSeam("Beta |u|=1/2", ctx, d, xs * f);
            }
            // B7 quantile: p = 1/2 reflection; p = 0.1 seed switch
            qSeam("Beta Q p=1/2", ctx, d, 0.5);
            qSeam("Beta Q p=0.1", ctx, d, 0.1);
            // t_lb < -700 shortcut: p where (log p + log a + lbeta)/a = -700
            const double lb = std::lgamma(a) + std::lgamma(b) - std::lgamma(a + b);
            const double lp = -700.0 * a - std::log(a) - lb;
            if (std::isfinite(lp) && lp > -744 && lp < -1)
                qSeam("Beta Q t_lb=-700", ctx, d, std::exp(lp));
        }
    // B4 a == b at x = 1/2
    for (double a : {1e-3, 0.3, 1.0, 2.5, 20.0, 1e3, 1e8}) {
        auto d = mk(a, a);
        cdfSeam("Beta a=b x=1/2", fmt("Beta(%g,%g)", a, a), d, 0.5);
    }
    // B3 parameter seams: a or b at 1e-3 (tiny), 1 (tail-integral), 20 (Stirling)
    for (double s0 : {1e-3, 1.0, 20.0}) {
        for (double other : {1e-4, 0.01, 0.5, 2.0, 30.0, 1e4}) {
            for (double x : {1e-6, 0.01, 0.3, 0.5, 0.7, 0.99, 1 - 1e-9}) {
                auto Fa = [&](double a) { return mk(a, other).getCumulativeProbability(x); };
                auto Fb = [&](double b) { return mk(other, b).getCumulativeProbability(x); };
                auto La = [&](double a) { return mk(a, other).getLogProbability(x); };
                const std::string ctx = fmt("Beta(*,%g)/(%g,*) x=%.17g", other, other, x);
                walk(fmt("Beta param a=%g CDF(a)", s0), ctx, Fa, s0, -1, true, {}, 48, 30);
                walk(fmt("Beta param b=%g CDF(b)", s0), ctx, Fb, s0, +1, true, {}, 48, 30);
                walk(fmt("Beta param a=%g logPDF(a)", s0), ctx, La, s0, 0, false, {}, 48, 30);
                for (double p : {0.3, 0.7}) {
                    if (x != 0.5)
                        continue;
                    auto Qa = [&](double a) { return mk(a, other).getQuantile(p); };
                    walk(fmt("Beta param a=%g Q(a)", s0), fmt("Beta(*,%g) p=%g", other, p), Qa, s0,
                         +1, false, {}, 48, 30);
                }
            }
        }
    }
    // B5 (a+b)^2 eps = min(a,b) switch in the Stirling prefactor
    for (double a : {20.0, 100.0, 1e4}) {
        const double bsw = std::sqrt(a / std::numeric_limits<double>::epsilon()) - a;
        for (double xf : {0.3, 1.0, 3.0}) {
            auto F = [&](double b) {
                return mk(a, b).getCumulativeProbability(xf * a / (a + bsw));
            };
            walk("Beta param (a+b)^2eps=min CDF(b)", fmt("Beta(%g,*~%.6g) x=%g*x0", a, bsw, xf), F,
                 bsw, +1, true, {}, 48, 30);
        }
    }
    report();
}

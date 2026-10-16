#include "seam.h"

int main() {
    // ---------------- Student-t
    for (double nu :
         {0.1, 0.5, 1.0, 2.0, 3.0, 7.0, 30.0, 39.999, 40.0, 40.001, 100.0, 1e4, 1e8, 1e12}) {
        auto d = StudentTDistribution::create(nu).unwrap();
        const std::string ctx = fmt("T(%.17g)", nu);
        cdfSeamC("T u=1 |t|=sqrt(nu)", ctx, d, -std::sqrt(nu), false);
        cdfSeamC("T u=1 |t|=sqrt(nu)", ctx, d, std::sqrt(nu), false);
        const double tsw = std::sqrt(1.5 * nu / (0.5 * nu + 1));
        cdfSeamC("T (a+1)u=1.5 tail/central", ctx, d, -tsw, false);
        cdfSeamC("T (a+1)u=1.5 tail/central", ctx, d, tsw, false);
        cdfSeam1("T t=0", ctx, d, 0.0, false);
        qSeamC("T Q p=1/4 central/tail", ctx, d, 0.25);
        qSeamC("T Q p=3/4 central/tail", ctx, d, 0.75);
        qSeam1("T Q p=1/2", ctx, d, 0.5);
        // logPDF overflow switch of t*t/nu
        const double tov = std::sqrt(std::numeric_limits<double>::max() * nu);
        if (std::isfinite(tov))
            walk(
                "T logPDF t^2/nu overflow", ctx, [&](double t) { return d.getLogProbability(t); },
                tov, 0, false);
    }
    for (double t : {-30.0, -6.0, -1.0, -0.3, 0.3, 1.0, 6.0, 30.0})
        walk(
            "T param nu=40 CDF(nu)", fmt("t=%g", t),
            [&](double nu) {
                return StudentTDistribution::create(nu).unwrap().getCumulativeProbability(t);
            },
            40.0, 0, true, {}, 48, 30);
    for (double p : {1e-10, 0.1, 0.3})
        walk(
            "T param nu=40 Q(nu)", fmt("p=%g", p),
            [&](double nu) { return StudentTDistribution::create(nu).unwrap().getQuantile(p); },
            40.0, +1, false, {}, 48, 30);

    // ---------------- Fisher F
    const std::vector<double> dfs = {1e-3, 0.1,  1.0,  1.999, 2.0, 2.001, 3.0,
                                     10.0, 39.9, 40.0, 100.0, 1e4, 1e8};
    for (double d1 : dfs)
        for (double d2 : dfs) {
            auto r = FDistribution::create(d1, d2);
            if (r.isError())
                continue;
            auto d = r.unwrap();
            const double a = d1 / 2, b = d2 / 2;
            const std::string ctx = fmt("F(%.17g,%.17g)", d1, d2);
            const double xb = d2 * (a + 1) / (d1 * (b + 1));
            cdfSeamC("F branch y=x_b", ctx, d, xb, true, 0, INFINITY);
            const double xm = bisectD(
                1e-300, 1e300, [&](double x) { return d.getCumulativeProbability(x) >= 0.5; });
            cdfSeamC("F CDF=1/2 orientation", ctx, d, xm, true, 0, INFINITY);
            cdfSeam1("F d1*x=DBL_MIN", ctx, d, std::numeric_limits<double>::min() / d1, true, 0,
                     INFINITY);
            cdfSeam1("F y=DBL_MIN extreme", ctx, d, std::numeric_limits<double>::min() * d2 / d1,
                     true, 0, INFINITY);
            cdfSeam1("F ybar=DBL_MIN extreme", ctx, d,
                     d2 / (d1 * std::numeric_limits<double>::min()), true, 0, INFINITY);
            qSeamC("F Q p=1/2", ctx, d, 0.5);
        }
    for (double d2 : {0.5, 3.0, 50.0})
        for (double x : {0.1, 1.0, 5.0}) {
            walk(
                "F param d1=2 CDF(d1)", fmt("F(*,%g) x=%g", d2, x),
                [&](double d1) {
                    return FDistribution::create(d1, d2).unwrap().getCumulativeProbability(x);
                },
                2.0, 0, true, {}, 48, 30);
            walk(
                "F param d2=2 CDF(d2)", fmt("F(%g,*) x=%g", d2, x),
                [&](double dd) {
                    return FDistribution::create(d2, dd).unwrap().getCumulativeProbability(x);
                },
                2.0, 0, true, {}, 48, 30);
            walk(
                "F param d1=40 CDF(d1)", fmt("F(*,%g) x=%g", d2, x),
                [&](double d1) {
                    return FDistribution::create(d1, d2 * 20).unwrap().getCumulativeProbability(x);
                },
                40.0, 0, true, {}, 48, 30);
        }

    // ---------------- LogNormal (normal_cdf z = 0; inverse_normal_cdf 0.075/0.15/0.85/0.925; s <
    // DBL_MIN)
    for (double mu : {0.0, 1.0, -3.0})
        for (double s : {0.1, 1.0, 3.0}) {
            auto d = LogNormalDistribution::create(mu, s).unwrap();
            const std::string ctx = fmt("LogN(%g,%g)", mu, s);
            cdfSeamC("LogN z=0", ctx, d, std::exp(mu), false, 0, INFINITY);
            for (double p : {0.075, 0.15, 0.5, 0.85, 0.925})
                qSeamC(fmt("LogN Q p=%g", p), ctx, d, p);
            qSeam1("LogN Q p=DBL_MIN", ctx, d, std::numeric_limits<double>::min());
        }
    // ---------------- Weibull (CDF -expm1, SF exp: F+S)
    for (double k : {0.3, 1.0, 2.0, 7.0}) {
        auto d = WeibullDistribution::create(k, 2.0).unwrap();
        cdfSeamC("Weibull x=scale", fmt("Weibull(%g,2)", k), d, 2.0, true, 0, INFINITY);
        cdfSeam1("Weibull x where F=1/2", fmt("Weibull(%g,2)", k), d,
                 2.0 * std::pow(std::log(2.0), 1 / k), true, 0, INFINITY);
        qSeamC("Weibull Q p=1/2", fmt("Weibull(%g,2)", k), d, 0.5);
    }
    // ---------------- Pareto (log ratio switch at x = 1e300 scale)
    for (double sc : {1e-300, 1e-10, 1.0})
        for (double al : {1e-3, 0.5, 3.0}) {
            auto d = ParetoDistribution::create(sc, al).unwrap();
            cdfSeam1("Pareto x=1e300*scale", fmt("Pareto(%g,%g)", sc, al), d, sc * 1e300, true, 0,
                     INFINITY);
            cdfSeam1("Pareto x=scale", fmt("Pareto(%g,%g)", sc, al), d, sc, true, sc, INFINITY);
        }
    // ---------------- digamma / trigamma / lbeta switches
    walk(
        "digamma x=10", "", [](double x) { return detail::digamma(x); }, 10.0, +1, false,
        [](double x) { return detail::trigamma(x); });
    walk("trigamma x=10", "", [](double x) { return detail::trigamma(x); }, 10.0, -1, false);
    for (double y : {1e-3, 0.5, 3.0, 19.0})
        walk(
            "lbeta max=20", fmt("lbeta(*,%g)", y), [&](double x) { return detail::lbeta(x, y); },
            20.0, -1, false);
    report();
}

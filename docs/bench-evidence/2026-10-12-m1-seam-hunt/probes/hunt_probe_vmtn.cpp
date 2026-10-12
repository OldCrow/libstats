#include "seam.h"

int main() {
    const double PI = 3.141592653589793;
    // ---------------- von Mises
    for (double k :
         {1e-18, 1e-17, 1e-3,  0.1,   0.25,   0.5,    1.0, 2.0,  10.0, 22.5, 30.0,         44.9,
          45.0,  60.0,  100.0, 999.0, 1000.0, 1001.0, 1e5, 1e10, 1e19, 1e20, 1.0000001e20, 1e25}) {
        for (double mu : {0.0, 1.0}) {
            auto d = VonMisesDistribution::create(mu, k).unwrap();
            const std::string ctx = fmt("VM(%g,%.17g)", mu, k);
            // central-mass/tail switch: tau = 2 asin(0.5/sqrt(k)) (2k s^2 = 1/2), tau = pi/2, F =
            // 1/4 or 3/4
            if (k >= 0.25) {
                const double tau = 2 * std::asin(0.5 / std::sqrt(k));
                cdfSeamC("VM 2ks^2=1/2", ctx, d, mu - tau, false, mu - 3.141592653589793,
                         mu + 3.141592653589793);
                cdfSeamC("VM 2ks^2=1/2", ctx, d, mu + tau, false, mu - 3.141592653589793,
                         mu + 3.141592653589793);
            }
            cdfSeamC("VM tau=pi/2 (also near_edge)", ctx, d, mu - PI / 2, false,
                     mu - 3.141592653589793, mu + 3.141592653589793);
            cdfSeamC("VM tau=pi/2 (also near_edge)", ctx, d, mu + PI / 2, false,
                     mu - 3.141592653589793, mu + 3.141592653589793);
            for (double target : {0.25, 0.75}) {
                const double x = bisectD(mu - PI, mu + PI, [&](double x) {
                    return d.getCumulativeProbability(x) >= target;
                });
                cdfSeamC("VM F=1/4,3/4 central-mass cap", ctx, d, x, false, mu - 3.141592653589793,
                         mu + 3.141592653589793);
            }
            // Taylor series A >= 45 / B >= 45 thresholds in t: cos^2(t/2) = 45/(2k)
            if (2 * k > 45) {
                const double t = 2 * std::acos(std::sqrt(45 / (2 * k)));
                cdfSeam1("VM tail A=45 / B=45", ctx, d, mu - t, false, mu - 3.141592653589793,
                         mu + 3.141592653589793);
                cdfSeam1("VM tail A=45 / B=45", ctx, d, mu - (PI - t), false,
                         mu - 3.141592653589793, mu + 3.141592653589793);
            }
            cdfSeam1("VM x=mu", ctx, d, mu, false, mu - 3.141592653589793, mu + 3.141592653589793);
            cdfSeam1("VM wrap x=mu+-pi", ctx, d, mu - PI, false, mu - 3.141592653589793,
                     mu + 3.141592653589793);
            cdfSeam1("VM wrap x=mu+-pi", ctx, d, mu + PI, false, mu - 3.141592653589793,
                     mu + 3.141592653589793);
            for (double p : {0.25, 0.5, 0.75})
                qSeamC(fmt("VM Q p=%g", p), ctx, d, p);
            qSeam1("VM Q p small", ctx, d, 1e-10);
        }
    }
    for (double k0 : {1e-17, 45.0, 1000.0, 1e20})
        for (double x : {-3.0, -1.0, -0.3, -0.01, 0.2, 2.0})
            walk(
                fmt("VM param kappa=%g CDF(k)", k0), fmt("x=%g", x),
                [&](double k) {
                    return VonMisesDistribution::create(0.0, k).unwrap().getCumulativeProbability(
                        x);
                },
                k0, x < 0 ? -1 : +1, true, {}, 48, 30);
    for (double k0 : {1e-17, 45.0, 1000.0, 1e20})
        for (double p : {1e-6, 0.1, 0.3})
            walk(
                fmt("VM param kappa=%g Q(k)", k0), fmt("p=%g", p),
                [&](double k) {
                    return VonMisesDistribution::create(0.0, k).unwrap().getQuantile(p);
                },
                k0, +1, false, {}, 48, 30);

    // ---------------- Truncated normal
    struct P {
        double mu, s, a, b;
    };
    std::vector<P> ps;
    for (double a : {-(double)INFINITY, -40.0, -5.0, -1.0, -1e-300, 0.0, 1e-300, 0.5, 3.0, 38.0})
        for (double w : {0.5, 3.0, (double)INFINITY})
            ps.push_back({0.0, 1.0, a, a + w});
    ps.push_back({0, 1, -50, -30});
    ps.push_back({0, 1, -40, -37});
    ps.push_back({0, 1, -INFINITY, -37.5});
    ps.push_back({2, 3, -1, 2});
    for (auto& q : ps) {
        auto r = TruncatedNormalDistribution::create(q.mu, q.s, q.a, q.b);
        if (r.isError())
            continue;
        auto d = r.unwrap();
        const std::string ctx = fmt("TN(%g,%g,%g,%g)", q.mu, q.s, q.a, q.b);
        if (q.mu > q.a && q.mu < q.b)
            cdfSeamC("TN xi=0 erf/normal_q (alpha<0)", ctx, d, q.mu, true, q.a, q.b);
        const double xm = bisectD(std::isfinite(q.a) ? q.a : -1e3, std::isfinite(q.b) ? q.b : 1e3,
                                  [&](double x) { return d.getCumulativeProbability(x) >= 0.5; });
        cdfSeamC("TN F=1/2", ctx, d, xm, true, q.a, q.b);
        // phi_xi < DBL_MIN switch: xi with Q(-xi) = DBL_MIN, about -37.5194
        const double xs = bisectD(-40.0, -30.0, [](double z) {
            return 0.5 * std::erfc(-z / std::sqrt(2.0)) >= std::numeric_limits<double>::min();
        });
        if (xs > q.a && xs < q.b)
            cdfSeam1("TN Phi(xi)=DBL_MIN", ctx, d, q.mu + q.s * xs, true, q.a, q.b);
        // quantile: q_low <= s_high crossover p* and p=1/2
        for (double p : {0.5, 0.25, 1e-10})
            qSeamC(fmt("TN Q p=%g", p), ctx, d, p);
        const double pa = 0.5 * std::erfc(-(q.a - q.mu) / q.s / std::sqrt(2.0));
        const double qb = 0.5 * std::erfc((q.b - q.mu) / q.s / std::sqrt(2.0));
        const double z = 1 - pa - qb;
        if (z > 0) {
            const double pstar = (qb - pa + z) / (2 * z);
            if (pstar > 0 && pstar < 1)
                qSeamC("TN Q q_low=s_high", ctx, d, pstar);
        }
    }
    // alpha = 0 parameter seam: lower bound a crossing mu
    for (double x : {-0.5, 0.0, 0.3, 1.0, 2.5})
        for (double b : {1.0, 3.0, (double)INFINITY}) {
            auto F = [&](double a) {
                auto r = TruncatedNormalDistribution::create(0.0, 1.0, a, b);
                return r.isError() ? NAN : r.unwrap().getCumulativeProbability(x);
            };
            walk("TN param a=mu (alpha=0) CDF(a)", fmt("TN(0,1,*,%g) x=%g", b, x), F, 0.0, -1, true,
                 {}, 48, 30, -INFINITY, std::min(x, b));
            walk("TN param a=mu (alpha=0) CDF(a) near0", fmt("TN(0,1,*,%g) x=%g", b, x), F, -1e-300,
                 -1, true, {}, 48, 30, -INFINITY, std::min(x, b));
        }
    report();
}

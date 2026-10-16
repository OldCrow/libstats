#include "seam.h"

template <class Make>
void paramSeam(const std::string& lab, const std::string& ctx, Make make, double k, double p0,
               int cdfDir, double lo, double hi) {
    walk(
        lab + " CDF", ctx,
        [&](double q) {
            auto r = make(q);
            return r.isError() ? NAN : r.unwrap().getCumulativeProbability(k);
        },
        p0, cdfDir, true, {}, 48, 30, lo, hi);
    walk(
        lab + " SF", ctx,
        [&](double q) {
            auto r = make(q);
            return r.isError() ? NAN : r.unwrap().getSurvival(k);
        },
        p0, -cdfDir, true, {}, 48, 30, lo, hi);
    fsCheck(
        lab + " F+S", ctx,
        [&](double q) {
            auto r = make(q);
            return r.isError() ? NAN : r.unwrap().getCumulativeProbability(k);
        },
        [&](double q) {
            auto r = make(q);
            return r.isError() ? NAN : r.unwrap().getSurvival(k);
        },
        p0, 16, 30);
}

int main() {
    // Poisson: CDF = Q(k+1, lambda), SF = P(k+1, lambda): seams in lambda at lambda = k + 2
    // (series/CF), the Temme zone edges (k + 1 >= 1e4), shape 20 (k = 19)
    auto mkP = [](double l) { return PoissonDistribution::create(l); };
    for (int k : {0, 1, 5, 18, 19, 20, 50, 1000, 9998, 9999, 10000, 100000}) {
        const double a = k + 1.0;
        paramSeam("Poisson lambda=k+2", fmt("Poisson k=%g", k), mkP, k, a + 1, -1, 1e-300, 1e300);
        paramSeam("Poisson lambda=k+2 [ctrl]", fmt("Poisson k=%g", k), mkP, k, (a + 1) * 1.003, -1,
                  1e-300, 1e300);
        if (a >= 1e4) {
            for (bool up : {true, false}) {
                double lo = up ? 0.0 : -0.99, hi = up ? 1.0 : 0.0;
                for (int i = 0; i < 200; ++i) {
                    double m = 0.5 * (lo + hi);
                    bool over = -2.0 * (std::log1p(m) - m) > 0.36;
                    if (up) {
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
                paramSeam("Poisson Temme |eta|=0.6", fmt("Poisson k=%g", k), mkP, k,
                          a * (1 + 0.5 * (lo + hi)), -1, 1e-300, 1e300);
            }
        }
    }
    // Binomial: CDF(k) = I_{1-p}(n-k, k+1): branch point in p, and Stirling at min(n-k, k+1) = 20
    auto mkB = [](int n) { return [n](double p) { return BinomialDistribution::create(n, p); }; };
    for (int n : {5, 30, 39, 40, 41, 100, 1000, 100000, 1000000000})
        for (int k : {0, 1, 5, 18, 19, 20, n / 2, n - 21, n - 20, n - 19, n - 2}) {
            if (k < 0 || k >= n)
                continue;
            const double xb = (n - k + 1.0) / (n + 3.0), pb = (k + 2.0) / (n + 3.0);
            const double p0 = pb <= 0.5 ? pb : 1.0 - xb;
            paramSeam("Binomial branch in p", fmt("Bin(%g) k=%g", n, k), mkB(n), k, p0, -1, 1e-300,
                      1 - 1e-16);
            paramSeam("Binomial branch in p [ctrl]", fmt("Bin(%g) k=%g", n, k), mkB(n), k,
                      p0 * 1.003, -1, 1e-300, 1 - 1e-16);
            // CDF = 1/2 in p
            auto B = mkB(n);
            const double pm = bisectD(1e-300, 1 - 1e-16, [&](double p) {
                return B(p).unwrap().getCumulativeProbability(k) < 0.5;
            });
            paramSeam("Binomial F=1/2 in p", fmt("Bin(%g) k=%g", n, k), mkB(n), k, pm, -1, 1e-300,
                      1 - 1e-16);
        }
    // NegBin: CDF = I_p(r, k+1), SF = I_{1-p}(k+1, r): r = 1 special path, branch point in p, r =
    // 20, r < 1 tail
    for (double r : {0.3, 0.999, 1.0, 1.001, 5.0, 19.9, 20.0, 50.0})
        for (int k : {0, 1, 5, 19, 20, 100, 10000}) {
            auto mk = [r](double p) { return NegativeBinomialDistribution::create(r, p); };
            const double pb = (r + 1) / (r + k + 3.0), ob = (k + 2.0) / (r + k + 3.0);
            const double p0 = pb <= 0.5 ? pb : 1 - ob;
            paramSeam("NegBin branch in p", fmt("NB(%g) k=%g", r, k), mk, k, p0, +1, 1e-300,
                      1 - 1e-16);
            paramSeam("NegBin branch in p [ctrl]", fmt("NB(%g) k=%g", r, k), mk, k, p0 * 1.003, +1,
                      1e-300, 1 - 1e-16);
            const double pm = bisectD(1e-12, 1 - 1e-12, [&](double p) {
                return mk(p).unwrap().getCumulativeProbability(k) < 0.5;
            });
            paramSeam("NegBin F=1/2 in p", fmt("NB(%g) k=%g", r, k), mk, k, pm, +1, 1e-300,
                      1 - 1e-16);
        }
    for (double r0 : {1.0, 20.0})
        for (double p : {0.01, 0.3, 0.7, 0.99})
            for (int k : {0, 3, 50}) {
                auto mk = [p](double r) { return NegativeBinomialDistribution::create(r, p); };
                paramSeam(fmt("NegBin param r=%g", r0), fmt("NB(*,%g) k=%g", p, k), mk, k, r0, -1,
                          1e-300, 1e300);
            }
    for (int n0 : {39, 40})
        (void)n0;
    report();
}

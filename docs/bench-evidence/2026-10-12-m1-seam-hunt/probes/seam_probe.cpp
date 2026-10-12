// Probe: are discrete quantiles monotone across p = 1/2, where the search switches from the CDF
// side to the survival side? Checks Q(p-) <= Q(1/2) <= Q(p+) for p- / p+ a few ulp to 1e-9 away,
// over dense parameter grids.
#define LIBSTATS_FULL_INTERFACE
#include "libstats/libstats.h"

#include <cmath>
#include <cstdio>
#include <functional>
#include <string>

using namespace stats;

static long checks = 0, bad = 0, rejected = 0, inexact = 0;
static double worstGap = 0.0;

template <typename D>
void probe(const std::string& what, const D& d) {
    const double half = 0.5;
    {  // can the seam misbehave here? only if cdf(k) + sf(k) != 1 at the median count
        const double k = d.getQuantile(half);
        for (double kk : {k - 1, k}) {
            const double gap = std::fabs(d.getCumulativeProbability(kk) + d.getSurvival(kk) - 1.0);
            if (gap > 0)
                ++inexact;
            if (gap > worstGap)
                worstGap = gap;
        }
    }
    const double ds[] = {0.0, 1e-16, 1e-15, 1e-13, 1e-11, 1e-9};
    double prev = -1e308;
    // walk p upward through the seam: every step must be non-decreasing
    for (int s = 5; s >= 1; --s) {
        const double p = half - ds[s];
        const double q = d.getQuantile(p);
        ++checks;
        if (q < prev) {
            ++bad;
            if (bad <= 100000)
                std::printf("NONMONO %s p=%.17g q=%.17g prev=%.17g\n", what.c_str(), p, q, prev);
        }
        prev = q;
    }
    double p = std::nextafter(half, 0.0);
    for (int i = 0; i < 3; ++i) {
        const double q = d.getQuantile(p);
        ++checks;
        if (q < prev) {
            ++bad;
            if (bad <= 100000)
                std::printf("NONMONO %s p=%.17g q=%.17g prev=%.17g\n", what.c_str(), p, q, prev);
        }
        prev = q;
        p = std::nextafter(p, 1.0);
    }
    for (int s = 1; s <= 5; ++s) {
        const double pp = half + ds[s];
        const double q = d.getQuantile(pp);
        ++checks;
        if (q < prev) {
            ++bad;
            if (bad <= 100000)
                std::printf("NONMONO %s p=%.17g q=%.17g prev=%.17g\n", what.c_str(), pp, q, prev);
        }
        prev = q;
    }
}

// Parameters where CDF(k) sits on 1/2 to the last bit: bisect a monotone parameter, then probe it
// and its 40 neighbouring doubles, where the CDF-side and survival-side rules can disagree.
template <typename Make>
void constructed(const char* name, int k, double lo, double hi, bool cdfDecreasesInParam,
                 Make make) {
    char buf[256];
    for (int it = 0; it < 200 && lo < hi; ++it) {
        const double mid = 0.5 * (lo + hi);
        if (mid == lo || mid == hi)
            break;
        auto r = make(mid);
        if (r.isError())
            return;
        const double c = r.unwrap().getCumulativeProbability(static_cast<double>(k));
        const bool above = c >= 0.5;
        if (above == cdfDecreasesInParam)
            lo = mid;
        else
            hi = mid;
    }
    double x = lo;
    for (int i = 0; i < 20; ++i)
        x = std::nextafter(x, -1e308);
    for (int i = 0; i < 41; ++i, x = std::nextafter(x, 1e308)) {
        auto r = make(x);
        if (r.isError()) {
            ++rejected;
            continue;
        }
        std::snprintf(buf, sizeof buf, "%s(k=%d, param=%.17g)", name, k, x);
        probe(buf, r.unwrap());
    }
}

int main() {
    char buf[256];
    for (int i = 0; i <= 4000; ++i) {  // Poisson lambda, log-spaced 1e-3 .. 1e7
        const double lam = std::pow(10.0, -3.0 + 10.0 * i / 4000.0);
        std::snprintf(buf, sizeof buf, "Poisson(%.17g)", lam);
        {
            auto r_ = PoissonDistribution::create(lam);
            if (r_.isError())
                ++rejected;
            else
                probe(buf, r_.unwrap());
        }
    }
    // Poisson at lambda where the median boundary sits close to 1/2: lambda = k + ln2/3 - ...
    for (int k = 1; k <= 3000; ++k) {
        for (double off : {-1e-3, -1e-6, 0.0, 1e-6, 1e-3}) {
            const double lam = k + std::log(2.0) - 1.0 / 3.0 + off;  // near-median lambda
            if (lam <= 0)
                continue;
            std::snprintf(buf, sizeof buf, "Poisson(%.17g)", lam);
            {
                auto r_ = PoissonDistribution::create(lam);
                if (r_.isError())
                    ++rejected;
                else
                    probe(buf, r_.unwrap());
            }
        }
    }
    for (int n : {1, 2, 3, 5, 10, 31, 100, 1001, 10000, 1000000}) {
        for (int j = 1; j < 400; ++j) {
            const double pr = j / 400.0;
            std::snprintf(buf, sizeof buf, "Binomial(%d,%.17g)", n, pr);
            {
                auto r_ = BinomialDistribution::create(n, pr);
                if (r_.isError())
                    ++rejected;
                else
                    probe(buf, r_.unwrap());
            }
        }
    }
    for (double r : {0.5, 1.0, 2.5, 10.0, 100.0, 1e4}) {
        for (int j = 1; j < 400; ++j) {
            const double pr = j / 400.0;
            std::snprintf(buf, sizeof buf, "NegBin(%g,%.17g)", r, pr);
            {
                auto r_ = NegativeBinomialDistribution::create(r, pr);
                if (r_.isError())
                    ++rejected;
                else
                    probe(buf, r_.unwrap());
            }
        }
    }
    for (int j = 1; j < 4000; ++j) {
        const double pr = j / 4000.0;
        std::snprintf(buf, sizeof buf, "Geometric(%.17g)", pr);
        {
            auto r_ = GeometricDistribution::create(pr);
            if (r_.isError())
                ++rejected;
            else
                probe(buf, r_.unwrap());
        }
    }
    for (int a = -5; a <= 5; ++a)
        for (int b = a; b <= a + 50; ++b) {
            std::snprintf(buf, sizeof buf, "Discrete(%d,%d)", a, b);
            {
                auto r_ = DiscreteDistribution::create(a, b);
                if (r_.isError())
                    ++rejected;
                else
                    probe(buf, r_.unwrap());
            }
        }
    for (int j = 1; j < 1000; ++j) {
        const double pr = j / 1000.0;
        std::snprintf(buf, sizeof buf, "Bernoulli(%.17g)", pr);
        {
            auto r_ = BernoulliDistribution::create(pr);
            if (r_.isError())
                ++rejected;
            else
                probe(buf, r_.unwrap());
        }
    }
    for (int k = 0; k <= 3000; ++k)
        constructed("PoissonC", k, 1e-6, 4000.0, true,
                    [](double l) { return PoissonDistribution::create(l); });
    for (int n : {3, 10, 31, 100, 1001, 10000})
        for (int k = 0; k < n; k += (n > 100 ? n / 97 : 1))
            constructed("BinomialC", k, 1e-12, 1 - 1e-12, true,
                        [n](double q) { return BinomialDistribution::create(n, q); });
    for (double r : {0.5, 2.5, 10.0, 100.0})
        for (int k = 0; k <= 400; ++k)
            constructed("NegBinC", k, 1e-9, 1 - 1e-9, false,
                        [r](double q) { return NegativeBinomialDistribution::create(r, q); });
    for (int k = 0; k <= 2000; ++k)
        constructed("GeometricC", k, 1e-9, 1 - 1e-9, false,
                    [](double q) { return GeometricDistribution::create(q); });
    std::printf(
        "checks %ld, non-monotone %ld, rejected params %ld; cdf+sf!=1 at the median count: %ld, "
        "worst gap %.3g\n",
        checks, bad, rejected, inexact, worstGap);
    return bad ? 1 : 0;
}

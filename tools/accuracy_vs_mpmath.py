#!/usr/bin/env python3
"""Compare libstats' accuracy_sweep output against mpmath (dps=50) reference
implementations, for issue #46 (accuracy characterization across all 19
libstats distributions).

This is the oracle/report half of #46: the accuracy_sweep C++ tool (owned by
a sibling change, tools/accuracy_sweep.cpp) enumerates (dist, method,
params, x, scalar_result, batch_result) rows to stdout/CSV; this script is
the independent reference and report generator that consumes that CSV.

INPUT CONTRACT -- one row per (dist, method, params, x) sample, comma
separated, no header required beyond the informational `#`-comment banner
the sweep tool emits (commit/isa/date). Columns:

    dist,method,p1_bits,p2_bits,x_bits,scalar_bits,batch_bits

  - All *_bits fields are 0x-hex uint64 IEEE-754 double bit patterns.
  - method in {pdf, logpdf, cdf, quantile}. For quantile, x_bits carries the
    probability p (not a domain point), and batch_bits may be the literal
    string "-" (no batch quantile path).
  - dist is the header basename (see DIST_PARAMS below for the parameter
    mapping used here, read from each include/libstats/distributions/<dist>.h
    Doxygen constructor comment).
  - p1_bits/p2_bits are the two constructor parameters in DECLARATION order;
    single-parameter distributions still carry a p2_bits field (its value is
    unused by the reference functions below).
  - Lines starting with `#` (the banner, and any trailing
    `# skipped_quantile ...` notes) are comments and are skipped.

ORACLE DOCTRINE (matches this repo's established generators -- see
scripts/gen_vonmises_cdf_vectors.py, scripts/gen_lognormal_cdf_vectors.py,
scripts/gen_gaussian_cdf_vectors.py, and tests/test_lognormal_cdf_accuracy.cpp's
law_budget comment, all read before writing this file):

  - mpmath at mp.dps = 50 (wider than the 40-digit CDF generators above,
    since this script also differentiates for pdf/logpdf and root-finds for
    quantile, both of which cost a few extra guard digits).
  - Gaussian/LogNormal CDF: erfc closed form, exactly as those two
    generators compute it (F = erfc(-z/sqrt(2))/2).
  - Von Mises CDF: direct quadrature of the density (independent of the
    library's Bessel-series/trapezoidal implementation), mu-centered wrap
    replicated in DOUBLE arithmetic exactly as gen_vonmises_cdf_vectors.py's
    wrap_to_pi -- reusing that file's PI_D anchoring: the wrap threshold
    must be the double-rounded value of pi (math.pi), NOT mpmath's wider
    mp.pi, or exact-endpoint rows spuriously disagree by a full branch-cut
    fold. See wrap_to_pi's docstring below for the copied rationale.
  - Discrete distributions (poisson, binomial, negative_binomial,
    geometric, discrete): pmf/cdf via mpmath's regularized incomplete
    gamma/beta identities where available (closed form), direct summation
    only where no closed form is simpler; quantile by monotone search
    against the closed-form CDF.
  - Deep-tail CDF rows (reference F < 1e-3) additionally report law_frac:
    the observed relative error as a fraction of the achievable-accuracy
    law rel(F) ~ |ln F| * 2^-52 established in
    tests/test_lognormal_cdf_accuracy.cpp's law_budget comment -- a flat
    relative-error budget is unachievable in the deep tail for ANY
    double-precision implementation of these closed forms, so a large
    max_rel alongside a small law_frac (<= ~1) is the law, not a defect.
  - NaN/Inf rows are contract-checked (NaN in -> NaN out, etc.) separately
    from the error statistics, never folded into max_abs/max_rel/p99_rel.

ORACLE SELF-CHECKS run unconditionally before any row is compared (see
`run_self_checks`) and this script exits non-zero if any fails -- a broken
oracle must never silently produce a report.

Usage:
    <python-with-mpmath> tools/accuracy_vs_mpmath.py <sweep.csv> [--out PATH]

Writes the stdout summary table, then rewrites the generated block of
docs/ACCURACY_CHARACTERIZATION.md (between the `<!-- BEGIN GENERATED -->`
and `<!-- END GENERATED -->` markers) with per-distribution tables built
from the same statistics.
"""

from __future__ import annotations

import argparse
import math
import os
import struct
import sys
from dataclasses import dataclass, field

import mpmath as mp

mp.mp.dps = 50

D = struct.Struct("<d")
Q = struct.Struct("<Q")

TWO_POW_NEG52 = mp.mpf(2) ** -52

# ---------------------------------------------------------------------------
# Bit <-> float helpers (matches the convention used by every gen_*.py
# generator in scripts/).
# ---------------------------------------------------------------------------


def bits_to_f64(b: int) -> float:
    return D.unpack(Q.pack(b))[0]


def f64_to_bits(x: float) -> int:
    return Q.unpack(D.pack(x))[0]


def parse_bits_field(s: str) -> "float | None":
    """Parse a 0x-hex uint64 bit pattern into a double, or None for the
    literal '-' (batch_bits sentinel for quantile rows with no batch path)."""
    s = s.strip()
    if s == "-":
        return None
    return bits_to_f64(int(s, 16))


# ---------------------------------------------------------------------------
# Parameter mapping: constructor parameter order per include/libstats/
# distributions/<dist>.h (read directly from each header's Doxygen comment
# before writing this table -- see the final report for any judgment calls).
# NPARAMS = 1 means p2_bits is present in the row but unused here.
# ---------------------------------------------------------------------------

DIST_PARAMS = {
    "gaussian": ("mean", "standardDeviation", 2),
    "lognormal": ("mu", "sigma", 2),
    "exponential": ("lambda", None, 1),
    "uniform": ("a", "b", 2),
    "poisson": ("lambda", None, 1),
    "gamma": ("alpha_shape", "beta_rate", 2),
    "discrete": ("a", "b", 2),  # integer bounds, discrete uniform on {a,...,b}
    "student_t": ("nu", None, 1),
    "cauchy": ("x0", "gamma_scale", 2),
    "von_mises": ("mu", "kappa", 2),
    "binomial": ("n", "p", 2),  # n is an integer trial count
    "negative_binomial": ("r", "p", 2),
    "geometric": ("p", None, 1),
    "beta": ("alpha", "beta", 2),
    "chi_squared": ("k", None, 1),
    "laplace": ("mu", "b_scale", 2),
    "pareto": ("scale_xm", "alpha", 2),
    "rayleigh": ("sigma", None, 1),
    "weibull": ("shape_k", "scale_lambda", 2),
}

DISCRETE_DISTS = {"poisson", "binomial", "negative_binomial", "geometric", "discrete"}

TWO_PI = 2 * mp.pi
SQRT2 = mp.sqrt(2)
LOG_2PI = mp.log(2 * mp.pi)


# ---------------------------------------------------------------------------
# Von Mises wrap: bit-exact DOUBLE replica of VonMisesDistribution::wrapAngle,
# copied (rationale included) from scripts/gen_vonmises_cdf_vectors.py. That
# generator's docstring explains at length why the wrap threshold must be
# the DOUBLE-rounded pi (math.pi, bit-identical to detail::PI) rather than
# mpmath's wider mp.pi: the library's own branch compare is a double compare
# against that same rounded constant, so anchoring the oracle's wrap on
# mp.pi instead would leave the exact double x = -PI_D just inside the
# interior of (-mp.pi, mp.pi] and disagree with the library's own fold-to-
# +PI_D convention at every exact-endpoint row -- a spurious ~1.0 CDF
# disagreement with nothing to do with kernel accuracy. This repo already
# litigated this seam (issue #51); reuse rather than rediscover it.
# ---------------------------------------------------------------------------

PI_D = math.pi
TWO_PI_D = 2.0 * math.pi


def wrap_to_pi(x: float, mu: float) -> "mp.mpf":
    t = math.fmod(x - mu, TWO_PI_D)
    if t <= -PI_D:
        t += TWO_PI_D
    if t > PI_D:
        t -= TWO_PI_D
    return mp.mpf(t)


_vm_denom_cache: dict = {}


def _vm_circle_integral(kappa: "mp.mpf", lo: "mp.mpf", hi: "mp.mpf") -> "mp.mpf":
    f = lambda p: mp.e ** (kappa * mp.cos(p))
    if lo <= 0 <= hi:
        return mp.quad(f, [lo, mp.mpf(0), hi])
    return mp.quad(f, [lo, hi])


def _vm_denom(kappa: "mp.mpf") -> "mp.mpf":
    key = float(kappa)
    if key not in _vm_denom_cache:
        _vm_denom_cache[key] = _vm_circle_integral(kappa, -mp.pi, mp.pi)
    return _vm_denom_cache[key]


def vonmises_cdf(mu: float, kappa: float, x: float) -> "mp.mpf":
    kap = mp.mpf(kappa)
    t = wrap_to_pi(x, mu)
    num = _vm_circle_integral(kap, -mp.pi, t)
    F = num / _vm_denom(kap)
    if F < 0:
        F = mp.mpf(0)
    if F > 1:
        F = mp.mpf(1)
    return F


def vonmises_quantile(mu: float, kappa: float, p: float) -> "mp.mpf":
    """Bisect t in [-pi, pi] on the mu=0-centered CDF for F(t) = p, then
    shift by mu and wrap. This mirrors the SHAPE of the library's own
    approach (build a mu=0 CDF, then shift/wrap for the actual mu) but not
    its 2049-point grid + linear interpolation, so scalar_bits should be
    expected to differ from this oracle by up to the grid's own
    interpolation error, not just double-precision rounding -- see the
    final report's judgment-call notes."""
    kap = mp.mpf(kappa)
    Z = _vm_denom(kap)
    pm = mp.mpf(p)

    def f(t):
        return _vm_circle_integral(kap, -mp.pi, t) / Z - pm

    if pm <= 0:
        t = -mp.pi
    elif pm >= 1:
        t = mp.pi
    else:
        t = mp.findroot(f, mp.mpf(0), solver="bisect", x0=(-mp.pi, mp.pi))
    result = mp.mpf(mu) + t
    # Wrap into (-pi, pi] at mpf precision (quantile has no library-side
    # bit-exact double wrap to replicate; this is a reference convention).
    while result > mp.pi:
        result -= TWO_PI
    while result <= -mp.pi:
        result += TWO_PI
    return result


# ---------------------------------------------------------------------------
# Reference implementations. Each entry: pdf(p1,p2,x), logpdf(p1,p2,x),
# cdf(p1,p2,x), quantile(p1,p2,p) -> mp.mpf. p1/p2 are passed as plain
# Python floats (the exact double values decoded from the row); each
# function lifts them to mpf itself.
# ---------------------------------------------------------------------------


def _gamma_cdf(alpha: "mp.mpf", beta: "mp.mpf", x: "mp.mpf") -> "mp.mpf":
    if x <= 0:
        return mp.mpf(0)
    return mp.gammainc(alpha, 0, beta * x, regularized=True)


def _gamma_quantile(alpha: float, beta: float, p: float) -> "mp.mpf":
    a = mp.mpf(alpha)
    b = mp.mpf(beta)
    pm = mp.mpf(p)
    if pm <= 0:
        return mp.mpf(0)
    if pm >= 1:
        return mp.inf
    # Wilson-Hilferty initial guess for a stable Newton start.
    z = mp.sqrt(2) * mp.erfinv(2 * pm - 1)
    guess = a * (1 - 1 / (9 * a) + z / (3 * mp.sqrt(a))) ** 3
    if guess <= 0 or not mp.isfinite(guess):
        guess = a / b
    x = mp.findroot(lambda xx: _gamma_cdf(a, b, xx) - pm, guess / b)
    return x if x > 0 else mp.mpf(0)


def _discrete_search_quantile(cdf_int, lo: int, hi: int, p: "mp.mpf") -> int:
    """Smallest integer k in [lo, hi] with cdf_int(k) >= p, by binary search
    (cdf_int assumed monotone non-decreasing on integers)."""
    if p <= 0:
        return lo
    while lo < hi:
        mid = (lo + hi) // 2
        if cdf_int(mid) >= p:
            hi = mid
        else:
            lo = mid + 1
    return lo


def _poisson_cdf(lam: "mp.mpf", k: int) -> "mp.mpf":
    if k < 0:
        return mp.mpf(0)
    return mp.gammainc(k + 1, lam, mp.inf, regularized=True)


def _binomial_cdf(n: int, p: "mp.mpf", k: int) -> "mp.mpf":
    if k < 0:
        return mp.mpf(0)
    if k >= n:
        return mp.mpf(1)
    return mp.betainc(n - k, k + 1, 0, 1 - p, regularized=True)


def _negbinom_cdf(r: "mp.mpf", p: "mp.mpf", k: int) -> "mp.mpf":
    if k < 0:
        return mp.mpf(0)
    return mp.betainc(r, k + 1, 0, p, regularized=True)


def _student_t_cdf(nu: "mp.mpf", t: "mp.mpf") -> "mp.mpf":
    if t == 0:
        return mp.mpf("0.5")
    xt = nu / (nu + t * t)
    ib = mp.betainc(nu / 2, mp.mpf("0.5"), 0, xt, regularized=True)
    return mp.mpf("0.5") * ib if t < 0 else 1 - mp.mpf("0.5") * ib


def _student_t_quantile(nu: "mp.mpf", p: "mp.mpf") -> "mp.mpf":
    if p == mp.mpf("0.5"):
        return mp.mpf(0)
    guess = SQRT2 * mp.erfinv(2 * p - 1)  # normal approx for the Newton start
    return mp.findroot(lambda t: _student_t_cdf(nu, t) - p, guess)


REFS: dict = {}


def _reg(name):
    def deco(cls):
        REFS[name] = cls
        return cls

    return deco


class Ref:
    """Base: subclasses set pdf/logpdf/cdf/quantile as staticmethods taking
    (p1, p2, x) -> mp.mpf (x is p for quantile)."""


@_reg("gaussian")
class GaussianRef(Ref):
    @staticmethod
    def pdf(mean, sigma, x):
        m, s = mp.mpf(mean), mp.mpf(sigma)
        z = (mp.mpf(x) - m) / s
        return mp.e ** (-z * z / 2) / (s * mp.sqrt(2 * mp.pi))

    @staticmethod
    def logpdf(mean, sigma, x):
        m, s = mp.mpf(mean), mp.mpf(sigma)
        z = (mp.mpf(x) - m) / s
        return -z * z / 2 - mp.log(s) - LOG_2PI / 2

    @staticmethod
    def cdf(mean, sigma, x):
        z = (mp.mpf(x) - mp.mpf(mean)) / mp.mpf(sigma)
        return mp.erfc(-z / SQRT2) / 2

    @staticmethod
    def quantile(mean, sigma, p):
        return mp.mpf(mean) + mp.mpf(sigma) * SQRT2 * mp.erfinv(2 * mp.mpf(p) - 1)


@_reg("lognormal")
class LogNormalRef(Ref):
    @staticmethod
    def pdf(mu, sigma, x):
        xm, mum, sm = mp.mpf(x), mp.mpf(mu), mp.mpf(sigma)
        if xm <= 0:
            return mp.mpf(0)
        z = (mp.log(xm) - mum) / sm
        return mp.e ** (-z * z / 2) / (xm * sm * mp.sqrt(2 * mp.pi))

    @staticmethod
    def logpdf(mu, sigma, x):
        xm, mum, sm = mp.mpf(x), mp.mpf(mu), mp.mpf(sigma)
        if xm <= 0:
            return -mp.inf
        z = (mp.log(xm) - mum) / sm
        return -z * z / 2 - mp.log(xm) - mp.log(sm) - LOG_2PI / 2

    @staticmethod
    def cdf(mu, sigma, x):
        xm, mum, sm = mp.mpf(x), mp.mpf(mu), mp.mpf(sigma)
        if xm <= 0:
            return mp.mpf(0)
        z = (mp.log(xm) - mum) / sm
        return mp.erfc(-z / SQRT2) / 2

    @staticmethod
    def quantile(mu, sigma, p):
        z = SQRT2 * mp.erfinv(2 * mp.mpf(p) - 1)
        return mp.e ** (mp.mpf(mu) + mp.mpf(sigma) * z)


@_reg("exponential")
class ExponentialRef(Ref):
    @staticmethod
    def pdf(lam, _p2, x):
        l, xm = mp.mpf(lam), mp.mpf(x)
        return l * mp.e ** (-l * xm) if xm >= 0 else mp.mpf(0)

    @staticmethod
    def logpdf(lam, _p2, x):
        l, xm = mp.mpf(lam), mp.mpf(x)
        return mp.log(l) - l * xm if xm >= 0 else -mp.inf

    @staticmethod
    def cdf(lam, _p2, x):
        l, xm = mp.mpf(lam), mp.mpf(x)
        return 1 - mp.e ** (-l * xm) if xm >= 0 else mp.mpf(0)

    @staticmethod
    def quantile(lam, _p2, p):
        return -mp.log(1 - mp.mpf(p)) / mp.mpf(lam)


@_reg("uniform")
class UniformRef(Ref):
    @staticmethod
    def pdf(a, b, x):
        a, b, xm = mp.mpf(a), mp.mpf(b), mp.mpf(x)
        return 1 / (b - a) if a <= xm <= b else mp.mpf(0)

    @staticmethod
    def logpdf(a, b, x):
        a, b, xm = mp.mpf(a), mp.mpf(b), mp.mpf(x)
        return -mp.log(b - a) if a <= xm <= b else -mp.inf

    @staticmethod
    def cdf(a, b, x):
        a, b, xm = mp.mpf(a), mp.mpf(b), mp.mpf(x)
        if xm < a:
            return mp.mpf(0)
        if xm > b:
            return mp.mpf(1)
        return (xm - a) / (b - a)

    @staticmethod
    def quantile(a, b, p):
        return mp.mpf(a) + mp.mpf(p) * (mp.mpf(b) - mp.mpf(a))


@_reg("gamma")
class GammaRef(Ref):
    @staticmethod
    def pdf(alpha, beta, x):
        a, b, xm = mp.mpf(alpha), mp.mpf(beta), mp.mpf(x)
        if xm <= 0:
            return mp.mpf(0)
        return b**a / mp.gamma(a) * xm ** (a - 1) * mp.e ** (-b * xm)

    @staticmethod
    def logpdf(alpha, beta, x):
        a, b, xm = mp.mpf(alpha), mp.mpf(beta), mp.mpf(x)
        if xm <= 0:
            return -mp.inf
        return a * mp.log(b) - mp.loggamma(a) + (a - 1) * mp.log(xm) - b * xm

    @staticmethod
    def cdf(alpha, beta, x):
        return _gamma_cdf(mp.mpf(alpha), mp.mpf(beta), mp.mpf(x))

    @staticmethod
    def quantile(alpha, beta, p):
        return _gamma_quantile(alpha, beta, p)


@_reg("discrete")
class DiscreteRef(Ref):
    @staticmethod
    def pdf(a, b, x):
        a, b = int(round(a)), int(round(b))
        xm = mp.mpf(x)
        n = b - a + 1
        return 1 / mp.mpf(n) if xm == mp.floor(xm) and a <= xm <= b else mp.mpf(0)

    @staticmethod
    def logpdf(a, b, x):
        p = DiscreteRef.pdf(a, b, x)
        return mp.log(p) if p > 0 else -mp.inf

    @staticmethod
    def cdf(a, b, x):
        a, b = int(round(a)), int(round(b))
        xm = mp.mpf(x)
        n = b - a + 1
        if xm < a:
            return mp.mpf(0)
        if xm >= b:
            return mp.mpf(1)
        return (mp.floor(xm) - a + 1) / mp.mpf(n)

    @staticmethod
    def quantile(a, b, p):
        a, b = int(round(a)), int(round(b))
        n = b - a + 1
        pm = mp.mpf(p)
        if pm <= 0:
            return mp.mpf(a)
        if pm >= 1:
            return mp.mpf(b)
        k = a - 1 + int(mp.ceil(pm * n))
        return mp.mpf(min(max(k, a), b))


@_reg("student_t")
class StudentTRef(Ref):
    @staticmethod
    def pdf(nu, _p2, x):
        n, xm = mp.mpf(nu), mp.mpf(x)
        return (
            mp.gamma((n + 1) / 2)
            / (mp.sqrt(n * mp.pi) * mp.gamma(n / 2))
            * (1 + xm * xm / n) ** (-(n + 1) / 2)
        )

    @staticmethod
    def logpdf(nu, _p2, x):
        n, xm = mp.mpf(nu), mp.mpf(x)
        return (
            mp.loggamma((n + 1) / 2)
            - mp.loggamma(n / 2)
            - mp.log(n * mp.pi) / 2
            - (n + 1) / 2 * mp.log(1 + xm * xm / n)
        )

    @staticmethod
    def cdf(nu, _p2, x):
        return _student_t_cdf(mp.mpf(nu), mp.mpf(x))

    @staticmethod
    def quantile(nu, _p2, p):
        return _student_t_quantile(mp.mpf(nu), mp.mpf(p))


@_reg("cauchy")
class CauchyRef(Ref):
    @staticmethod
    def pdf(x0, gamma, x):
        x0, g, xm = mp.mpf(x0), mp.mpf(gamma), mp.mpf(x)
        z = (xm - x0) / g
        return 1 / (mp.pi * g * (1 + z * z))

    @staticmethod
    def logpdf(x0, gamma, x):
        x0, g, xm = mp.mpf(x0), mp.mpf(gamma), mp.mpf(x)
        z = (xm - x0) / g
        return -mp.log(mp.pi * g) - mp.log1p(z * z)

    @staticmethod
    def cdf(x0, gamma, x):
        x0, g, xm = mp.mpf(x0), mp.mpf(gamma), mp.mpf(x)
        return mp.mpf("0.5") + mp.atan((xm - x0) / g) / mp.pi

    @staticmethod
    def quantile(x0, gamma, p):
        return mp.mpf(x0) + mp.mpf(gamma) * mp.tan(mp.pi * (mp.mpf(p) - mp.mpf("0.5")))


@_reg("von_mises")
class VonMisesRef(Ref):
    @staticmethod
    def pdf(mu, kappa, x):
        m, k, xm = mp.mpf(mu), mp.mpf(kappa), mp.mpf(x)
        return mp.e ** (k * mp.cos(xm - m)) / (2 * mp.pi * mp.besseli(0, k))

    @staticmethod
    def logpdf(mu, kappa, x):
        m, k, xm = mp.mpf(mu), mp.mpf(kappa), mp.mpf(x)
        return k * mp.cos(xm - m) - mp.log(2 * mp.pi * mp.besseli(0, k))

    @staticmethod
    def cdf(mu, kappa, x):
        return vonmises_cdf(mu, kappa, x)

    @staticmethod
    def quantile(mu, kappa, p):
        return vonmises_quantile(mu, kappa, p)


@_reg("binomial")
class BinomialRef(Ref):
    @staticmethod
    def pdf(n, p, x):
        n = int(round(n))
        k = int(round(x))
        pm = mp.mpf(p)
        if k < 0 or k > n:
            return mp.mpf(0)
        return mp.binomial(n, k) * pm**k * (1 - pm) ** (n - k)

    @staticmethod
    def logpdf(n, p, x):
        v = BinomialRef.pdf(n, p, x)
        return mp.log(v) if v > 0 else -mp.inf

    @staticmethod
    def cdf(n, p, x):
        n = int(round(n))
        k = int(math.floor(x))
        return _binomial_cdf(n, mp.mpf(p), k)

    @staticmethod
    def quantile(n, p, prob):
        n = int(round(n))
        pm = mp.mpf(p)
        pr = mp.mpf(prob)
        return mp.mpf(_discrete_search_quantile(lambda k: _binomial_cdf(n, pm, k), 0, n, pr))


@_reg("negative_binomial")
class NegativeBinomialRef(Ref):
    @staticmethod
    def pdf(r, p, x):
        r, pm = mp.mpf(r), mp.mpf(p)
        k = int(round(x))
        if k < 0:
            return mp.mpf(0)
        return mp.gamma(k + r) / (mp.gamma(r) * mp.factorial(k)) * pm**r * (1 - pm) ** k

    @staticmethod
    def logpdf(r, p, x):
        v = NegativeBinomialRef.pdf(r, p, x)
        return mp.log(v) if v > 0 else -mp.inf

    @staticmethod
    def cdf(r, p, x):
        k = int(math.floor(x))
        return _negbinom_cdf(mp.mpf(r), mp.mpf(p), k)

    @staticmethod
    def quantile(r, p, prob):
        rm, pm, pr = mp.mpf(r), mp.mpf(p), mp.mpf(prob)
        # Doubling search for an upper bound, then binary search (no fixed
        # support ceiling for negative binomial's failure count).
        hi = 1
        while _negbinom_cdf(rm, pm, hi) < pr:
            hi *= 2
            if hi > 1 << 40:
                break
        return mp.mpf(_discrete_search_quantile(lambda k: _negbinom_cdf(rm, pm, k), 0, hi, pr))


@_reg("geometric")
class GeometricRef(Ref):
    @staticmethod
    def pdf(p, _p2, x):
        pm = mp.mpf(p)
        k = int(round(x))
        return pm * (1 - pm) ** k if k >= 0 else mp.mpf(0)

    @staticmethod
    def logpdf(p, _p2, x):
        pm = mp.mpf(p)
        k = int(round(x))
        return mp.log(pm) + k * mp.log1p(-pm) if k >= 0 else -mp.inf

    @staticmethod
    def cdf(p, _p2, x):
        pm = mp.mpf(p)
        k = int(math.floor(x))
        if k < 0:
            return mp.mpf(0)
        return 1 - (1 - pm) ** (k + 1)

    @staticmethod
    def quantile(p, _p2, prob):
        pm, pr = mp.mpf(p), mp.mpf(prob)
        if pr <= 0:
            return mp.mpf(0)
        if pm >= 1:
            return mp.mpf(0)
        k = mp.ceil(mp.log1p(-pr) / mp.log1p(-pm) - 1)
        k = max(k, 0)
        # Closed form can be off by one at exact boundaries; nudge via the
        # exact CDF (monotone, so a local search of +/-2 always corrects it).
        while GeometricRef.cdf(p, _p2, k - 1) >= pr and k > 0:
            k -= 1
        while GeometricRef.cdf(p, _p2, k) < pr:
            k += 1
        return mp.mpf(k)


@_reg("beta")
class BetaRef(Ref):
    @staticmethod
    def pdf(alpha, beta, x):
        a, b, xm = mp.mpf(alpha), mp.mpf(beta), mp.mpf(x)
        if xm <= 0 or xm >= 1:
            return mp.mpf(0)
        return xm ** (a - 1) * (1 - xm) ** (b - 1) / mp.beta(a, b)

    @staticmethod
    def logpdf(alpha, beta, x):
        a, b, xm = mp.mpf(alpha), mp.mpf(beta), mp.mpf(x)
        if xm <= 0 or xm >= 1:
            return -mp.inf
        return (a - 1) * mp.log(xm) + (b - 1) * mp.log1p(-xm) - mp.log(mp.beta(a, b))

    @staticmethod
    def cdf(alpha, beta, x):
        a, b, xm = mp.mpf(alpha), mp.mpf(beta), mp.mpf(x)
        if xm <= 0:
            return mp.mpf(0)
        if xm >= 1:
            return mp.mpf(1)
        return mp.betainc(a, b, 0, xm, regularized=True)

    @staticmethod
    def quantile(alpha, beta, p):
        a, b, pm = mp.mpf(alpha), mp.mpf(beta), mp.mpf(p)
        if pm <= 0:
            return mp.mpf(0)
        if pm >= 1:
            return mp.mpf(1)
        guess = a / (a + b)
        return mp.findroot(lambda xx: mp.betainc(a, b, 0, xx, regularized=True) - pm, guess)


@_reg("chi_squared")
class ChiSquaredRef(Ref):
    @staticmethod
    def pdf(k, _p2, x):
        kk, xm = mp.mpf(k), mp.mpf(x)
        if xm <= 0:
            return mp.mpf(0)
        return xm ** (kk / 2 - 1) * mp.e ** (-xm / 2) / (2 ** (kk / 2) * mp.gamma(kk / 2))

    @staticmethod
    def logpdf(k, _p2, x):
        kk, xm = mp.mpf(k), mp.mpf(x)
        if xm <= 0:
            return -mp.inf
        return (kk / 2 - 1) * mp.log(xm) - xm / 2 - (kk / 2) * mp.log(2) - mp.loggamma(kk / 2)

    @staticmethod
    def cdf(k, _p2, x):
        return _gamma_cdf(mp.mpf(k) / 2, mp.mpf("0.5"), mp.mpf(x))

    @staticmethod
    def quantile(k, _p2, p):
        return _gamma_quantile(float(k) / 2.0, 0.5, p)


@_reg("laplace")
class LaplaceRef(Ref):
    @staticmethod
    def pdf(mu, b, x):
        m, bb, xm = mp.mpf(mu), mp.mpf(b), mp.mpf(x)
        return mp.e ** (-abs(xm - m) / bb) / (2 * bb)

    @staticmethod
    def logpdf(mu, b, x):
        m, bb, xm = mp.mpf(mu), mp.mpf(b), mp.mpf(x)
        return -abs(xm - m) / bb - mp.log(2 * bb)

    @staticmethod
    def cdf(mu, b, x):
        m, bb, xm = mp.mpf(mu), mp.mpf(b), mp.mpf(x)
        if xm < m:
            return mp.mpf("0.5") * mp.e ** ((xm - m) / bb)
        return 1 - mp.mpf("0.5") * mp.e ** (-(xm - m) / bb)

    @staticmethod
    def quantile(mu, b, p):
        m, bb, pm = mp.mpf(mu), mp.mpf(b), mp.mpf(p)
        if pm < mp.mpf("0.5"):
            return m + bb * mp.log(2 * pm)
        return m - bb * mp.log(2 * (1 - pm))


@_reg("pareto")
class ParetoRef(Ref):
    @staticmethod
    def pdf(scale, alpha, x):
        xm0, a, xm = mp.mpf(scale), mp.mpf(alpha), mp.mpf(x)
        if xm < xm0:
            return mp.mpf(0)
        return a * xm0**a / xm ** (a + 1)

    @staticmethod
    def logpdf(scale, alpha, x):
        xm0, a, xm = mp.mpf(scale), mp.mpf(alpha), mp.mpf(x)
        if xm < xm0:
            return -mp.inf
        return mp.log(a) + a * mp.log(xm0) - (a + 1) * mp.log(xm)

    @staticmethod
    def cdf(scale, alpha, x):
        xm0, a, xm = mp.mpf(scale), mp.mpf(alpha), mp.mpf(x)
        if xm < xm0:
            return mp.mpf(0)
        return 1 - (xm0 / xm) ** a

    @staticmethod
    def quantile(scale, alpha, p):
        xm0, a, pm = mp.mpf(scale), mp.mpf(alpha), mp.mpf(p)
        return xm0 / (1 - pm) ** (1 / a)


@_reg("rayleigh")
class RayleighRef(Ref):
    @staticmethod
    def pdf(sigma, _p2, x):
        s, xm = mp.mpf(sigma), mp.mpf(x)
        if xm < 0:
            return mp.mpf(0)
        return xm / s**2 * mp.e ** (-xm * xm / (2 * s * s))

    @staticmethod
    def logpdf(sigma, _p2, x):
        s, xm = mp.mpf(sigma), mp.mpf(x)
        if xm <= 0:
            return -mp.inf
        return mp.log(xm) - 2 * mp.log(s) - xm * xm / (2 * s * s)

    @staticmethod
    def cdf(sigma, _p2, x):
        s, xm = mp.mpf(sigma), mp.mpf(x)
        if xm < 0:
            return mp.mpf(0)
        return 1 - mp.e ** (-xm * xm / (2 * s * s))

    @staticmethod
    def quantile(sigma, _p2, p):
        s, pm = mp.mpf(sigma), mp.mpf(p)
        return s * mp.sqrt(-2 * mp.log(1 - pm))


@_reg("weibull")
class WeibullRef(Ref):
    @staticmethod
    def pdf(shape, scale, x):
        k, lam, xm = mp.mpf(shape), mp.mpf(scale), mp.mpf(x)
        if xm < 0:
            return mp.mpf(0)
        return (k / lam) * (xm / lam) ** (k - 1) * mp.e ** (-((xm / lam) ** k))

    @staticmethod
    def logpdf(shape, scale, x):
        k, lam, xm = mp.mpf(shape), mp.mpf(scale), mp.mpf(x)
        if xm <= 0:
            return -mp.inf
        return mp.log(k) - mp.log(lam) + (k - 1) * (mp.log(xm) - mp.log(lam)) - (xm / lam) ** k

    @staticmethod
    def cdf(shape, scale, x):
        k, lam, xm = mp.mpf(shape), mp.mpf(scale), mp.mpf(x)
        if xm < 0:
            return mp.mpf(0)
        return 1 - mp.e ** (-((xm / lam) ** k))

    @staticmethod
    def quantile(shape, scale, p):
        k, lam, pm = mp.mpf(shape), mp.mpf(scale), mp.mpf(p)
        return lam * (-mp.log(1 - pm)) ** (1 / k)


@_reg("poisson")
class PoissonRef(Ref):
    @staticmethod
    def pdf(lam, _p2, x):
        l = mp.mpf(lam)
        k = int(round(x))
        if k < 0:
            return mp.mpf(0)
        return mp.e ** (-l) * l**k / mp.factorial(k)

    @staticmethod
    def logpdf(lam, _p2, x):
        l = mp.mpf(lam)
        k = int(round(x))
        if k < 0:
            return -mp.inf
        return -l + k * mp.log(l) - mp.loggamma(k + 1)

    @staticmethod
    def cdf(lam, _p2, x):
        k = int(math.floor(x))
        return _poisson_cdf(mp.mpf(lam), k)

    @staticmethod
    def quantile(lam, _p2, p):
        l, pr = mp.mpf(lam), mp.mpf(p)
        hi = max(1, int(float(l) * 2) + 10)
        while _poisson_cdf(l, hi) < pr:
            hi *= 2
            if hi > 1 << 40:
                break
        return mp.mpf(_discrete_search_quantile(lambda k: _poisson_cdf(l, k), 0, hi, pr))


# ---------------------------------------------------------------------------
# Oracle self-checks. Run before any comparison; assertion failure -> the
# process exits non-zero (see main()). At least two independent known-value
# checks per distribution: a median/symmetry identity and a closed-form
# spot value, matching this repo's generator doctrine (house rule: "generated
# references are trusted over comments" -- so the CHECKS, not comments, are
# what's trusted here).
# ---------------------------------------------------------------------------


def run_self_checks() -> list:
    """Returns the list of (name, ok, detail) tuples; raises AssertionError
    immediately on the first failing check (fail-fast, matches the
    generators' doctrine)."""
    results = []

    def check(name, cond, detail=""):
        assert cond, f"SELF-CHECK FAILED: {name} {detail}"
        results.append((name, True, detail))

    tol = mp.mpf("1e-30")

    # --- gaussian ---
    check("gaussian.cdf(mean)==0.5", abs(GaussianRef.cdf(0.3, 2.0, 0.3) - mp.mpf("0.5")) <= tol)
    check(
        "gaussian.symmetry",
        abs(GaussianRef.cdf(0.0, 1.0, 1.5) - (1 - GaussianRef.cdf(0.0, 1.0, -1.5))) <= tol,
    )

    # --- lognormal ---
    check(
        "lognormal.cdf(exp(mu))==0.5",
        abs(LogNormalRef.cdf(0.5, 1.0, float(mp.e ** mp.mpf("0.5"))) - mp.mpf("0.5")) <= mp.mpf("1e-15"),
    )
    check(
        "lognormal.pdf_positive_support",
        LogNormalRef.pdf(0.0, 1.0, -1.0) == 0 and LogNormalRef.pdf(0.0, 1.0, 1.0) > 0,
    )

    # --- exponential ---
    check(
        "exponential.cdf(1/lambda)==1-e^-1",
        abs(ExponentialRef.cdf(2.0, 0, 0.5) - (1 - mp.e ** -1)) <= tol,
    )
    check("exponential.quantile_inverts_cdf", abs(ExponentialRef.quantile(3.0, 0, ExponentialRef.cdf(3.0, 0, 0.7))
                                                    - mp.mpf(0.7)) <= mp.mpf("1e-15"))

    # --- uniform ---
    check("uniform.cdf(mid)==0.5", UniformRef.cdf(2.0, 6.0, 4.0) == mp.mpf("0.5"))
    check("uniform.pdf_outside_zero", UniformRef.pdf(0.0, 1.0, 5.0) == 0)

    # --- poisson ---
    check("poisson.pmf(0;lambda)==e^-lambda", abs(PoissonRef.pdf(2.5, 0, 0) - mp.e ** mp.mpf("-2.5")) <= tol)
    check(
        "poisson.cdf_sums_to_pmf",
        abs(PoissonRef.cdf(3.0, 0, 2) - sum(PoissonRef.pdf(3.0, 0, k) for k in range(3))) <= tol,
    )

    # --- gamma ---
    check(
        "gamma.cdf(0)==0_and_cdf(inf)->1",
        GammaRef.cdf(2.0, 1.0, 0.0) == 0 and GammaRef.cdf(2.0, 1.0, 1e6) > mp.mpf("0.999999"),
    )
    check(
        "gamma.quantile_inverts_cdf",
        abs(GammaRef.quantile(3.0, 2.0, GammaRef.cdf(3.0, 2.0, 1.5)) - mp.mpf(1.5)) <= mp.mpf("1e-15"),
    )

    # --- discrete ---
    check("discrete.pdf_uniform", DiscreteRef.pdf(2, 5, 3) == mp.mpf(1) / 4)
    check("discrete.cdf(b)==1", DiscreteRef.cdf(2, 5, 5) == 1)

    # --- student_t ---
    check("student_t.cdf(0)==0.5", StudentTRef.cdf(5.0, 0, 0.0) == mp.mpf("0.5"))
    check(
        "student_t.symmetry",
        abs(StudentTRef.cdf(5.0, 0, 1.2) - (1 - StudentTRef.cdf(5.0, 0, -1.2))) <= tol,
    )

    # --- cauchy ---
    check("cauchy.cdf(x0)==0.5", CauchyRef.cdf(1.0, 2.0, 1.0) == mp.mpf("0.5"))
    check(
        "cauchy.quantile_inverts_cdf",
        abs(CauchyRef.quantile(0.0, 1.0, CauchyRef.cdf(0.0, 1.0, 3.0)) - mp.mpf(3.0)) <= mp.mpf("1e-15"),
    )

    # --- von_mises ---
    check(
        "vonmises.cdf(mu)==0.5",
        abs(VonMisesRef.cdf(0.7, 2.0, 0.7) - mp.mpf("0.5")) <= mp.mpf("1e-25"),
    )
    check(
        "vonmises.kappa0_uniform_closed_form",
        abs(VonMisesRef.cdf(0.0, 1e-12, 1.0) - (mp.mpf(1.0) + mp.pi) / TWO_PI) <= mp.mpf("1e-6"),
    )

    # --- binomial ---
    check("binomial.pmf_sums_to_one", abs(sum(BinomialRef.pdf(5, 0.3, k) for k in range(6)) - 1) <= tol)
    check("binomial.cdf(n)==1", BinomialRef.cdf(5, 0.3, 5) == 1)

    # --- negative_binomial ---
    check(
        "negbinom.pmf(0;r,p)==p^r",
        abs(NegativeBinomialRef.pdf(3.0, 0.4, 0) - mp.mpf(0.4) ** 3) <= tol,
    )
    check(
        "negbinom.cdf_monotone",
        NegativeBinomialRef.cdf(3.0, 0.4, 5) >= NegativeBinomialRef.cdf(3.0, 0.4, 2),
    )

    # --- geometric ---
    check("geometric.pmf(0)==p", GeometricRef.pdf(0.3, 0, 0) == mp.mpf(0.3))
    check(
        "geometric.cdf_closed_form",
        abs(GeometricRef.cdf(0.3, 0, 4) - (1 - (1 - mp.mpf(0.3)) ** 5)) <= tol,
    )

    # --- beta ---
    check("beta.cdf(0)==0_cdf(1)==1", BetaRef.cdf(2.0, 3.0, 0.0) == 0 and BetaRef.cdf(2.0, 3.0, 1.0) == 1)
    check(
        "beta.symmetric_alpha_eq_beta",
        abs(BetaRef.cdf(2.0, 2.0, 0.5) - mp.mpf("0.5")) <= tol,
    )

    # --- chi_squared ---
    check(
        "chisq.equals_gamma_k_over_2_half",
        abs(ChiSquaredRef.cdf(4.0, 0, 3.0) - GammaRef.cdf(2.0, 0.5, 3.0)) <= tol,
    )
    check("chisq.cdf(0)==0", ChiSquaredRef.cdf(4.0, 0, 0.0) == 0)

    # --- laplace ---
    check("laplace.cdf(mu)==0.5", LaplaceRef.cdf(1.0, 2.0, 1.0) == mp.mpf("0.5"))
    check(
        "laplace.symmetry",
        abs(LaplaceRef.cdf(0.0, 1.0, 2.0) - (1 - LaplaceRef.cdf(0.0, 1.0, -2.0))) <= tol,
    )

    # --- pareto ---
    check("pareto.cdf(scale)==0", ParetoRef.cdf(1.0, 3.0, 1.0) == 0)
    check(
        "pareto.quantile_inverts_cdf",
        abs(ParetoRef.quantile(1.0, 3.0, ParetoRef.cdf(1.0, 3.0, 2.5)) - mp.mpf(2.5)) <= mp.mpf("1e-15"),
    )

    # --- rayleigh ---
    check(
        "rayleigh.cdf_closed_form",
        abs(RayleighRef.cdf(2.0, 0, 2.0) - (1 - mp.e ** mp.mpf("-0.5"))) <= tol,
    )
    check("rayleigh.cdf(0)==0", RayleighRef.cdf(2.0, 0, 0.0) == 0)

    # --- weibull ---
    check(
        "weibull.reduces_to_exponential_at_shape1",
        abs(WeibullRef.cdf(1.0, 2.0, 3.0) - ExponentialRef.cdf(0.5, 0, 3.0)) <= tol,
    )
    check("weibull.cdf(0)==0", WeibullRef.cdf(2.0, 1.0, 0.0) == 0)

    return results


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------


@dataclass
class Row:
    dist: str
    method: str
    p1: float
    p2: float
    x: float  # for quantile rows, this is p
    scalar: float
    batch: "float | None"
    lineno: int


def parse_csv(path: str) -> "tuple[list[Row], dict]":
    rows: list[Row] = []
    meta: dict = {}
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                if "commit=" in line and "isa=" in line:
                    meta["banner"] = line.lstrip("#").strip()
                continue
            if line == "dist,method,p1_bits,p2_bits,x_bits,scalar_bits,batch_bits":
                # Literal column-header line the sweep tool emits (see the
                # input contract in this file's module docstring) -- not a
                # `#`-comment, but not data either.
                continue
            parts = line.split(",")
            if len(parts) != 7:
                raise ValueError(f"{path}:{lineno}: expected 7 fields, got {len(parts)}: {line!r}")
            dist, method, p1s, p2s, xs, scs, bts = parts
            p1 = parse_bits_field(p1s)
            p2 = parse_bits_field(p2s)
            x = parse_bits_field(xs)
            scalar = parse_bits_field(scs)
            batch = parse_bits_field(bts)
            if p1 is None or x is None or scalar is None:
                raise ValueError(f"{path}:{lineno}: p1/x/scalar must not be '-': {line!r}")
            rows.append(
                Row(dist.strip(), method.strip(), p1, p2 if p2 is not None else 0.0, x, scalar, batch, lineno)
            )
    return rows, meta


# ---------------------------------------------------------------------------
# Comparison / statistics
# ---------------------------------------------------------------------------


def law_of_f(f_ref: "mp.mpf") -> "mp.mpf":
    """Achievable double-precision relative-error law in the CDF's deep
    tail: rel(F) ~ |ln F| * 2^-52. See tests/test_lognormal_cdf_accuracy.cpp's
    law_budget comment -- this is the UNPADDED law (that test's pinned
    budget adds >2x headroom on top of this for gating; this report uses
    the bare law as the yardstick for "law-limited, not a defect")."""
    if f_ref <= 0:
        return TWO_POW_NEG52 * mp.mpf(20)
    return abs(mp.log(f_ref)) * TWO_POW_NEG52


def relerr(got: "mp.mpf", ref: "mp.mpf") -> "mp.mpf":
    if abs(ref) < mp.mpf("1e-290"):
        return abs(got - ref)
    return abs(got - ref) / abs(ref)


@dataclass
class SourceStats:
    n: int = 0
    max_abs: "mp.mpf" = field(default_factory=lambda: mp.mpf(0))
    max_rel: "mp.mpf" = field(default_factory=lambda: mp.mpf(0))
    rels: list = field(default_factory=list)
    worst_x: float = 0.0
    max_law_frac: "mp.mpf" = field(default_factory=lambda: mp.mpf(0))
    law_worst_x: float = 0.0
    has_law_rows: bool = False

    def observe(self, got: "mp.mpf", ref: "mp.mpf", x: float, is_cdf_tail: bool):
        a = abs(got - ref)
        r = relerr(got, ref)
        self.n += 1
        self.rels.append(r)
        if a > self.max_abs:
            self.max_abs = a
        if r > self.max_rel:
            self.max_rel = r
            self.worst_x = x
        if is_cdf_tail:
            self.has_law_rows = True
            lf = r / law_of_f(ref)
            if lf > self.max_law_frac:
                self.max_law_frac = lf
                self.law_worst_x = x

    def p99_rel(self) -> "mp.mpf":
        if not self.rels:
            return mp.mpf(0)
        s = sorted(self.rels)
        idx = min(len(s) - 1, int(math.ceil(0.99 * len(s))) - 1)
        return s[max(idx, 0)]


@dataclass
class GroupResult:
    dist: str
    method: str
    scalar: SourceStats = field(default_factory=SourceStats)
    batch: SourceStats = field(default_factory=SourceStats)
    batch_vs_scalar_max_abs: "mp.mpf" = field(default_factory=lambda: mp.mpf(0))
    batch_vs_scalar_max_rel: "mp.mpf" = field(default_factory=lambda: mp.mpf(0))
    n_batch_rows: int = 0
    violations: list = field(default_factory=list)


def is_nan(v: float) -> bool:
    return isinstance(v, float) and math.isnan(v)


def compare(rows: list) -> "tuple[dict, list]":
    groups: dict = {}
    skipped = []

    for row in rows:
        ref_fns = REFS.get(row.dist)
        if ref_fns is None:
            skipped.append((row.lineno, f"unknown dist {row.dist!r}"))
            continue
        fn = getattr(ref_fns, row.method, None)
        if fn is None:
            skipped.append((row.lineno, f"unknown method {row.method!r} for {row.dist}"))
            continue

        key = (row.dist, row.method)
        g = groups.setdefault(key, GroupResult(row.dist, row.method))

        nan_input = is_nan(row.p1) or is_nan(row.p2) or is_nan(row.x)
        inf_input = math.isinf(row.x)

        if nan_input:
            # Contract check only: NaN in -> NaN out, both paths.
            if not is_nan(row.scalar):
                g.violations.append((row.lineno, "scalar", "NaN input did not produce NaN scalar output"))
            if row.batch is not None and not is_nan(row.batch):
                g.violations.append((row.lineno, "batch", "NaN input did not produce NaN batch output"))
            continue

        try:
            ref = fn(row.p1, row.p2, row.x)
        except (ValueError, ZeroDivisionError, OverflowError) as exc:
            if inf_input:
                skipped.append((row.lineno, f"oracle skipped inf-input row ({exc})"))
                continue
            raise

        if not mp.isfinite(ref):
            # Reference itself is +-inf (e.g. quantile at p=1 for unbounded
            # support): contract-check finiteness/sign agreement rather than
            # a relative error, which is not meaningful against an infinite
            # reference.
            scalar_ok = math.isinf(row.scalar) and (row.scalar > 0) == (ref > 0)
            if not scalar_ok:
                g.violations.append(
                    (row.lineno, "scalar", f"reference is {ref}, scalar_bits decoded to {row.scalar}")
                )
            continue

        is_cdf_tail = row.method == "cdf" and ref < mp.mpf("1e-3")
        g.scalar.observe(mp.mpf(row.scalar), ref, row.x, is_cdf_tail)

        if row.batch is not None:
            g.n_batch_rows += 1
            g.batch.observe(mp.mpf(row.batch), ref, row.x, is_cdf_tail)
            bs_abs = abs(mp.mpf(row.batch) - mp.mpf(row.scalar))
            bs_rel = relerr(mp.mpf(row.batch), mp.mpf(row.scalar))
            g.batch_vs_scalar_max_abs = max(g.batch_vs_scalar_max_abs, bs_abs)
            g.batch_vs_scalar_max_rel = max(g.batch_vs_scalar_max_rel, bs_rel)

    return groups, skipped


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def fmt(x) -> str:
    if isinstance(x, mp.mpf):
        if x == 0:
            return "0"
        return mp.nstr(x, 4, strip_zeros=True)
    return str(x)


def print_stdout_summary(groups: dict, skipped: list, violations_total: int) -> None:
    print(f"{'dist':<18} {'method':<9} {'src':<6} {'n':>5} {'max_abs':>12} {'max_rel':>12} "
          f"{'p99_rel':>12} {'law_frac':>10} {'worst_x':>14}")
    for (dist, method), g in sorted(groups.items()):
        for src_name, s in (("scalar", g.scalar), ("batch", g.batch)):
            if s.n == 0:
                continue
            law_col = fmt(s.max_law_frac) if s.has_law_rows else "-"
            print(
                f"{dist:<18} {method:<9} {src_name:<6} {s.n:>5} {fmt(s.max_abs):>12} "
                f"{fmt(s.max_rel):>12} {fmt(s.p99_rel()):>12} {law_col:>10} {s.worst_x:>14.6g}"
            )
        if g.n_batch_rows:
            print(
                f"{'':<18} {'':<9} {'b_vs_s':<6} {g.n_batch_rows:>5} "
                f"{fmt(g.batch_vs_scalar_max_abs):>12} {fmt(g.batch_vs_scalar_max_rel):>12}"
            )
        if g.violations:
            for lineno, src, msg in g.violations:
                print(f"  VIOLATION line {lineno} ({src}): {msg}")

    if skipped:
        print(f"\n{len(skipped)} row(s) skipped:")
        for lineno, msg in skipped[:20]:
            print(f"  line {lineno}: {msg}")
        if len(skipped) > 20:
            print(f"  ... and {len(skipped) - 20} more")

    print(f"\ntotal contract violations: {violations_total}")


def render_markdown(groups: dict) -> str:
    lines = []
    dists = sorted({d for d, _ in groups.keys()})
    for dist in dists:
        lines.append(f"### {dist}\n")
        lines.append("| method | source | max_abs | max_rel | p99_rel | law_frac(cdf) | batch_vs_scalar | worst_x |")
        lines.append("|---|---|---|---|---|---|---|---|")
        methods = sorted(m for d, m in groups.keys() if d == dist)
        for method in methods:
            g = groups[(dist, method)]
            for src_name, s in (("scalar", g.scalar), ("batch", g.batch)):
                if s.n == 0:
                    continue
                law_col = fmt(s.max_law_frac) if s.has_law_rows else "-"
                bvs = (
                    f"abs={fmt(g.batch_vs_scalar_max_abs)}, rel={fmt(g.batch_vs_scalar_max_rel)}"
                    if src_name == "batch" and g.n_batch_rows
                    else "-"
                )
                lines.append(
                    f"| {method} | {src_name} | {fmt(s.max_abs)} | {fmt(s.max_rel)} | "
                    f"{fmt(s.p99_rel())} | {law_col} | {bvs} | {s.worst_x:.6g} |"
                )
            if g.violations:
                lines.append(
                    f"| {method} | *(contract)* | {len(g.violations)} violation(s) -- see stdout log | | | | | |"
                )
        lines.append("")
    return "\n".join(lines)


GEN_BEGIN = "<!-- BEGIN GENERATED -->"
GEN_END = "<!-- END GENERATED -->"


def rewrite_doc(doc_path: str, generated_body: str) -> None:
    with open(doc_path, "r", encoding="utf-8") as f:
        text = f.read()
    if GEN_BEGIN not in text or GEN_END not in text:
        raise ValueError(f"{doc_path}: missing {GEN_BEGIN}/{GEN_END} markers")
    pre, rest = text.split(GEN_BEGIN, 1)
    _, post = rest.split(GEN_END, 1)
    new_text = pre + GEN_BEGIN + "\n\n" + generated_body + "\n" + GEN_END + post
    with open(doc_path, "w", encoding="utf-8") as f:
        f.write(new_text)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", help="path to the accuracy_sweep CSV output")
    parser.add_argument(
        "--out",
        default=None,
        help="path to docs/ACCURACY_CHARACTERIZATION.md (default: alongside this script's repo root)",
    )
    args = parser.parse_args(argv)

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    doc_path = args.out or os.path.join(repo_root, "docs", "ACCURACY_CHARACTERIZATION.md")

    print("Running oracle self-checks (mp.dps=50)...")
    try:
        checks = run_self_checks()
    except AssertionError as exc:
        print(f"ORACLE SELF-CHECK FAILED: {exc}", file=sys.stderr)
        return 1
    print(f"  {len(checks)} self-checks passed.\n")

    rows, meta = parse_csv(args.csv)
    if "banner" in meta:
        print(f"Input banner: {meta['banner']}")
    print(f"Parsed {len(rows)} rows from {args.csv}\n")

    groups, skipped = compare(rows)
    violations_total = sum(len(g.violations) for g in groups.values())

    print_stdout_summary(groups, skipped, violations_total)

    body = render_markdown(groups)
    rewrite_doc(doc_path, body)
    print(f"\nWrote generated block to {doc_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

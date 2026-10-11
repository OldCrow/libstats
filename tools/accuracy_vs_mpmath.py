#!/usr/bin/env python3
"""Compare libstats' accuracy_sweep output against mpmath (dps=50) reference
implementations, for issue #46 (accuracy characterization across all 27
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
    unused by the reference functions below). `truncated_normal` is the one
    distribution with MORE than two constructor parameters: p1/p2 carry its
    first two in declaration order (mean, standardDeviation) and the
    truncation window is recovered from _TN_WINDOWS, keyed on that pair --
    widening the CSV was not an option, since p3/p4 columns would rewrite all
    6063 pre-v2.4.0 rows the characterization baseline is measured against.
    The sweep tool's instances therefore carry pairwise-distinct
    (mean, sd), and an unrecognised pair raises rather than guessing.
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
  - Relative error is scored down to DBL_MIN. Below it the error is the ULP
    distance in subnormal units (2^-1074), reported on the relative scale as
    ulps * 2^-52, and a reference at or below half the smallest subnormal is
    an exact 0 (see `relerr`). Until 2026-10-11 any reference below 1e-290
    was scored by ABSOLUTE error, so a subnormal or tiny result could be
    wrong in every digit and still score ~1e-300.
  - Every fallback from a helper's primary method is counted and printed
    after the summary (`FALLBACKS`); a method that fails its own convergence
    test raises `OracleNonConvergence` and stops the run.

ORACLE SELF-CHECKS run unconditionally before any row is compared (see
`run_self_checks`) and this script exits non-zero if any fails -- a broken
oracle must never silently produce a report. They include negative controls:
a 1-ULP perturbation must score as at least half an ULP, and the large-shape
paths are anchored against an independent quadrature.

Usage:
    <python-with-mpmath> tools/accuracy_vs_mpmath.py <sweep.csv> [--out PATH]

Writes the stdout summary table, then rewrites the generated block of
docs/ACCURACY_CHARACTERIZATION.md (between the `<!-- BEGIN GENERATED -->`
and `<!-- END GENERATED -->` markers) with per-distribution tables built
from the same statistics.
"""

from __future__ import annotations

import argparse
import collections
import re
import math
import os
import struct
import sys
import time
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
    # --- v2.4.0 (#54/#55/#56/#57) ---
    "logistic": ("mu", "s_scale", 2),
    "gumbel": ("mu", "beta_scale", 2),
    "erlang": ("k_int_shape", "lambda_rate", 2),
    "fisher_f": ("d1", "d2", 2),
    "inverse_gamma": ("alpha_shape", "beta_scale", 2),  # SCALE, not a rate
    "half_normal": ("sigma", None, 1),
    # 4 constructor parameters; p1/p2 are the first two and _TN_WINDOWS
    # supplies (lowerBound, upperBound) -- see the input contract above.
    "truncated_normal": ("mean", "standardDeviation", 2),
    "bernoulli": ("p", None, 1),
}

DISCRETE_DISTS = {
    "poisson", "binomial", "negative_binomial", "geometric", "discrete",
    "bernoulli",
}

# Truncation windows for the sweep tool's truncated_normal instances,
# keyed on the (mean, standardDeviation) pair the CSV carries. The other half
# of the contract lives in tools/accuracy_sweep.cpp's sweepContinuous4 comment;
# both sides refuse to guess, so the tables cannot drift apart silently.
_TN_WINDOWS = {
    (0.0, 1.0): (-2.0, 2.0),                          # symmetric, straddling
    (0.0, 2.0): (10.0, 12.0),                         # same-tail, alpha=5 beta=6
    (1000.0, 100.0): (float("-inf"), float("inf")),   # degenerate -> Gaussian
    # Half-infinite windows (switch-band instances, 2026-10-11), standardized
    # bounds in the comments.
    (0.0, 1.5): (7.5, float("inf")),                  # [5, inf)
    (0.0, 0.5): (15.0, float("inf")),                 # [30, inf)
    (4.0, 1.0): (2.0, float("inf")),                  # [-2, inf), straddling
    (1.0, 1.0): (float("-inf"), -4.0),                # (-inf, -5]
    (2.0, 1.0): (float("-inf"), -28.0),               # (-inf, -30]
    (3.0, 1.0): (float("-inf"), 5.0),                 # (-inf, 2], straddling
    (0.0, 0.25): (float("-inf"), 0.0),                # (-inf, 0]
}


def _tn_window(mean: float, sd: float) -> "tuple[float, float]":
    key = (float(mean), float(sd))
    if key not in _TN_WINDOWS:
        raise RuntimeError(
            f"truncated_normal instance (mean={mean!r}, sd={sd!r}) has no window "
            f"in _TN_WINDOWS. The sweep tool added or changed an instance without "
            f"updating this table; known keys: {sorted(_TN_WINDOWS)}"
        )
    return _TN_WINDOWS[key]

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


_vm_log_norm_cache: dict = {}


def _vm_log_norm(kappa: float) -> "mp.mpf":
    """log of int_{-pi}^{pi} exp(kappa (cos s - 1)) ds = log(2 pi I0(kappa)) - kappa,
    at 80 digits. The e^-kappa scaling keeps it finite at kappa = 1e8."""
    if kappa not in _vm_log_norm_cache:
        with mp.workdps(80):
            k = mp.mpf(kappa)
            _vm_log_norm_cache[kappa] = mp.log(2 * mp.pi) + mp.log(mp.besseli(0, k)) - k
    return _vm_log_norm_cache[kappa]


def _vm_small_side(kappa: float, u: "mp.mpf", exact: bool = True) -> "mp.mpf":
    """Probability mass within u of the antimode, on one side: int_0^u of the mu = 0
    density at s = v - pi, i.e. F(-pi + u) = 1 - F(pi - u) for 0 <= u <= pi.

    Integrated in the antimode distance u, never in s = -pi + u: near the antimode the
    sum -pi + u keeps no digit of a u below ~1e-80, and kappa = 1e-6 puts the p = 1e-300
    quantile at u ~ 6e-300.

    Until 2026-10-11 the CDF was the quotient of two unscaled quadratures of
    e^{kappa cos s}. mp.quad's convergence test is ABSOLUTE, so a tail integral far below the
    peak's scale stopped at the lowest degree; _quad_log_tail integrates the integrand scaled
    to 1 at its maximum instead.

    exact=False lets a CDF row skip the quadrature where even the bound u * density(u) (the
    integrand increases on [0, u]) rounds to 0 in double: kappa = 1e8 reaches e^-8.6e7.
    """
    u = mp.mpf(u)
    if u <= 0:
        return mp.mpf(0)
    if kappa == 0:
        return u / (2 * mp.pi)
    with mp.workdps(80):
        k = mp.mpf(kappa)
        lz = _vm_log_norm(kappa)
        g = lambda v: -k * (1 + mp.cos(v)) - lz  # increasing on [0, pi]
        uu = min(u, +mp.pi)
        if not exact and g(uu) + mp.log(uu) < _LOG_HALF_DENORM - 1:
            r = mp.exp(g(uu) + mp.log(uu))
        else:
            sc = 1 / mp.sqrt(k) if k > 1 else mp.mpf(1)
            r = _quad_log_tail(g, uu, True, 0, mp.pi, sc)
    return +r


def _vm_log_density_u(kappa: float, u: "mp.mpf") -> "mp.mpf":
    """log of the mu = 0 density at antimode distance u (the integrand of _vm_small_side)."""
    with mp.workdps(80):
        r = -mp.mpf(kappa) * (1 + mp.cos(mp.mpf(u))) - _vm_log_norm(kappa)
    return +r


def _vm_cdf_t(kappa: float, t: "mp.mpf") -> "mp.mpf":
    """mu = 0 CDF at t in [-pi, pi], from whichever side of the mode is smaller."""
    t = mp.mpf(t)
    with mp.workdps(80):
        if t <= 0:
            F = _vm_small_side(kappa, t + mp.pi, exact=False)
        else:
            F = 1 - _vm_small_side(kappa, mp.pi - t, exact=False)
    F = +F
    return min(max(F, mp.mpf(0)), mp.mpf(1))


def vonmises_cdf(mu: float, kappa: float, x: float) -> "mp.mpf":
    return _vm_cdf_t(kappa, wrap_to_pi(x, mu))


def vonmises_quantile(mu: float, kappa: float, p: float) -> "mp.mpf":
    """Solve the mu = 0 CDF for F(t) = p on the small side (antimode distance u, see
    _vm_small_side), then shift by mu and wrap. This mirrors the SHAPE of the library's
    own approach (build a mu=0 CDF, then shift/wrap for the actual mu)."""
    pm = mp.mpf(p)
    if pm <= 0:
        t = -mp.pi
    elif pm >= 1:
        t = mp.pi
    elif pm == mp.mpf("0.5"):
        # The median is mu by symmetry; a solver returns quadrature noise (2.5e-30 at
        # kappa = 1), which the exact-zero contract check then flags.
        t = mp.mpf(0)
    else:
        lower = pm < mp.mpf("0.5")
        target = pm if lower else 1 - pm
        # Bracket: the density is at most ~sqrt(kappa) + 1, so u below
        # target / (sqrt(kappa) + 1) holds less than target.
        u_lo = target / (mp.sqrt(mp.mpf(kappa)) + 1) / 8
        u = _newton_log_root(lambda uu: _vm_small_side(kappa, uu),
                             lambda uu: _vm_log_density_u(kappa, uu),
                             target, u_lo, +mp.pi, mp.pi / 2)
        t = -mp.pi + u if lower else mp.pi - u
    if mu == 0:
        # t is in (-pi, pi) by construction; wrapping it again would fold a
        # -pi + u whose u is below dps 50 (kappa = 1e-6, p = 1e-300) to +pi.
        return t
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


# ---------------------------------------------------------------------------
# Fallback accounting and convergence failures (2026-10-11). A helper that
# leaves its primary method counts the event in FALLBACKS, which main() prints
# after the summary, so no reference is built by a path the reader cannot
# see. A method that fails its own convergence test raises
# OracleNonConvergence instead of returning its last iterate.
# ---------------------------------------------------------------------------

FALLBACKS: "collections.Counter[str]" = collections.Counter()


class OracleNonConvergence(RuntimeError):
    """A reference method failed its own convergence test."""


def _note_fallback(kind: str) -> None:
    FALLBACKS[kind] += 1


# log(2^-1075): a value below exp of this rounds to 0 in double, so a guard
# that returns a leading term there cannot change any scored reference.
_LOG_HALF_DENORM = -1075 * mp.log(2)


def _gamma_lower_lead(alpha, y):
    """(lead, bound) for P(alpha, y), y < alpha + 1: P = e^lead * M with
    1 <= M <= (alpha + 1)/(alpha + 1 - y), so lead <= log P <= bound."""
    lead = alpha * mp.log(y) - y - mp.log(alpha) - mp.loggamma(alpha)
    return lead, lead + mp.log((alpha + 1) / (alpha + 1 - y))


def _gamma_upper_lead(alpha, y):
    """(lead, bound) for Q(alpha, y), y > alpha - 1: Q = e^lead * U with
    1 <= U <= y/(y - alpha + 1) for alpha >= 1, so lead <= log Q <= bound."""
    lead = (alpha - 1) * mp.log(y) - y - mp.loggamma(alpha)
    return lead, lead + mp.log(y / (y - alpha + 1))


def _betainc_lead(a, b, x):
    """(lead, bound) for I_x(a, b) = x^a/(a B(a, b)) * 2F1(a, 1 - b; a + 1; x):
    the 2F1 factor is a int_0^1 t^(a-1) (1 - x t)^(b-1) dt, at most 1 for
    b >= 1 and at most (1 - x)^(b-1) for b < 1."""
    lead = a * mp.log(x) - mp.log(a) - (mp.loggamma(a) + mp.loggamma(b) - mp.loggamma(a + b))
    return lead, lead + ((b - 1) * mp.log1p(-x) if b < 1 else 0)


def _quad_log_tail(g, x, lower, lo, hi, scale):
    """int exp(g(t)) dt over [lo, x] (lower) or [x, hi] (upper), for g unimodal
    with x on the side away from its mode, so exp(g) peaks at x on the
    integrated interval.

    mp.quad's convergence test is ABSOLUTE: integrating exp(g) directly, a
    tail integral of 1e-185 stops at the lowest degree with a 1e-11 relative
    error. The integrand here is exp(g - g(x)), which is 1 at x, and exp(g(x))
    is applied afterwards. Breakpoints walk away from x by doubling from 1/8
    of the local decay length min(scale, 1/|g'(x)|) until g has fallen 230
    below g(x) (e^-230 ~ 1e-100, past dps 50) or the interval ends. Callers
    run it at elevated precision (workdps 80).
    """
    x = mp.mpf(x)
    gx = g(x)
    if not mp.isfinite(gx):
        raise OracleNonConvergence(f"_quad_log_tail: g(x) = {gx} at x = {x}")
    try:
        slope = abs(mp.diff(g, x))
    except (ValueError, ZeroDivisionError):
        slope = mp.mpf(0)
    h = mp.mpf(scale)
    if mp.isfinite(slope) and slope > 0 and 1 / slope < h:
        h = 1 / slope
    h /= 8
    pts = [x]
    d = h
    for _ in range(4000):
        p = x - d if lower else x + d
        if lower and p <= lo:
            pts.append(mp.mpf(lo))
            break
        if (not lower) and p >= hi:
            pts.append(mp.mpf(hi))
            break
        pts.append(p)
        if g(p) < gx - 230:
            break
        d *= 2
    else:
        raise OracleNonConvergence(f"_quad_log_tail: no decay within 4000 doublings from x = {x}")
    pts.sort()
    # Integrate over s in [0, 1] (t = t0 + L s): mp.quad's error floor is
    # absolute, so the unit interval keeps it relative when the walk is short
    # (an interval of width 2e-60 otherwise meets a 1e-82 floor).
    t0, L = pts[0], pts[-1] - pts[0]
    spts = [(p - t0) / L for p in pts]
    val, err = mp.quad(lambda sv: mp.exp(g(t0 + L * sv) - gx), spts, error=True)
    if not (val > 0 and err <= val * mp.mpf("1e-40")):
        raise OracleNonConvergence(
            f"_quad_log_tail: quadrature error estimate {mp.nstr(err, 3)} against "
            f"{mp.nstr(val, 3)} at x = {mp.nstr(x, 17)}")
    return mp.exp(gx) * L * val


def _gamma_pq_quad(a, y):
    """(P(a, y), Q(a, y)) by quadrature of the gamma density, the small side
    integrated directly (the other as its complement). Independent of mpmath's
    gammainc: the primary path for shapes above GAMMA_QUAD_SHAPE, the fallback
    below it, and the anchor the self-checks compare gammainc against."""
    with mp.workdps(80):
        a = mp.mpf(a)
        y = mp.mpf(y)
        if y <= 0:
            P, Q = mp.mpf(0), mp.mpf(1)
        else:
            lga = mp.loggamma(a)
            g = lambda t: (a - 1) * mp.log(t) - t - lga if t > 0 else mp.mpf("-inf")
            if y <= a - 1:
                P = _quad_log_tail(g, y, True, 0, mp.inf, mp.sqrt(a))
                Q = 1 - P
            else:
                Q = _quad_log_tail(g, y, False, 0, mp.inf, mp.sqrt(max(a, mp.mpf(1))))
                P = 1 - Q
    return +P, +Q


def _student_t_cdf_quad(nu, t):
    """Student-t CDF by quadrature of the density, the small side directly."""
    with mp.workdps(80):
        n = mp.mpf(nu)
        tt = mp.mpf(t)
        C = mp.loggamma((n + 1) / 2) - mp.loggamma(n / 2) - mp.log(n * mp.pi) / 2
        g = lambda u: C - (n + 1) / 2 * mp.log1p(u * u / n)
        if tt <= 0:
            F = _quad_log_tail(g, tt, True, -mp.inf, mp.inf, mp.mpf(1))
        else:
            F = 1 - _quad_log_tail(g, tt, False, -mp.inf, mp.inf, mp.mpf(1))
    return +F


def _newton_log_root(fn, log_dfn, target, lo, hi, x0, increasing=True):
    """Root of fn(x) = target for fn monotone and positive on (lo, hi),
    0 < lo < x0 < hi, by Newton on log fn against y = log x (slope
    +-x exp(log_dfn(x)) / fn(x), log_dfn the log of |fn'|), inside a bracket
    in y kept by sign, bisecting whenever a step leaves it. The root must lie
    in (lo, hi); a run that cannot converge raises. In (log x, log fn) a power-law tail is a line
    and an exponential tail a parabola, so a handful of evaluations suffice
    where each costs a quadrature (von Mises, ~0.3 s); false position took
    ~100 on the large-kappa tail.
    """
    ylo, yhi, y = mp.log(lo), mp.log(hi), mp.log(x0)
    ltarget = mp.log(target)
    tol = mp.mpf("1e-45")
    for _ in range(300):
        x = mp.exp(y)
        v = fn(x)
        if v <= 0:
            if increasing:
                ylo = y
            else:
                yhi = y
            y = (ylo + yhi) / 2
            continue
        h = mp.log(v) - ltarget
        if h == 0:
            return x
        if (h < 0) == increasing:
            ylo = y
        else:
            yhi = y
        slope = x * mp.exp(log_dfn(x) - mp.log(v))
        step = h / (slope if increasing else -slope)
        if abs(step) <= tol:
            return mp.exp(y - step)
        y_new = y - step
        if not (ylo < y_new < yhi):
            y_new = (ylo + yhi) / 2
        if yhi - ylo <= tol:
            return mp.exp(y_new)
        y = y_new
    raise OracleNonConvergence(f"_newton_log_root: no convergence (target {mp.nstr(target, 6)})")


def _tail_logspace_bisect(fn, target, seed, increasing=True, hi_cap=None, step0=4096,
                          max_steps=600):
    """Root of fn(x) = target for fn strictly monotone on (0, hi_cap or inf),
    by geometric-midpoint (log-space) bisection at mpf precision.

    Replaces the secant mp.findroot calls originally used for the gamma /
    student-t / beta quantile oracles. A secant step that lands in a
    deep-tail plateau -- where fn is constant to 50 digits -- stalls there
    permanently (observed on the real #46 sweep: gamma quantile near
    p = 1 - 1e-16 stuck at |fn - target| ~ 1e-15 against a 2.6e-54
    tolerance). Why not pure log-space bisection (the first replacement):
    correct, but a fixed ~200 fn evaluations per row is hours of wall time
    when fn is an incomplete beta with a huge first parameter -- student-t
    nu = 1e6 costs tens of ms PER EVALUATION at dps 50, and minutes at the
    extreme arguments the expansion probes. The hybrid below keeps
    bisection's bracket invariant (cannot stall; anti-stagnation forcing
    bounds it at ~2x bisection worst case) while false position in
    (log x, log fn) coordinates converges superlinearly -- exactly, in one
    step, wherever the tail is a power law, since F ~ C*x^a is a straight
    line in those coordinates. Typical rows finish in 10-25 evaluations.
    """
    seed = mp.mpf(seed)
    if (not mp.isfinite(seed)) or seed <= 0:
        seed = mp.mpf(1)
    if hi_cap is not None and seed >= hi_cap:
        seed = mp.mpf(hi_cap) / 2
    sgn = 1 if increasing else -1
    # Bracket expansion doubles the step each miss (with an exponent cap so
    # mpf exponents stay small ints): a FIXED step cannot reach the bracket
    # for small shape parameters -- gamma alpha=0.01 at p ~ 1e-300 has its
    # quantile near 1e-30000, ~2800 fixed /4096 steps away but only ~13
    # doubling steps. Log-space bisection is indifferent to the overshoot:
    # it converges in log-width, which the doubling only grows linearly.
    #
    # step0 is the first expansion factor: 4096 for a rough seed, close to 1
    # for an asymptotic seed accurate to a few digits (large-shape gamma,
    # large-nu Student-t), where a 4096x bracket would waste evaluations that
    # each cost a quadrature. Squaring recovers from a seed that is wrong.
    step_cap = mp.mpf(2) ** 1000000
    lo = seed
    step = mp.mpf(step0)
    for _ in range(200):
        if sgn * (fn(lo) - target) < 0:
            break
        lo /= step
        step = min(step * step, step_cap)
    else:
        raise RuntimeError("quantile bracket expansion failed (low side)")
    if hi_cap is not None:
        hi = mp.mpf(hi_cap)
    else:
        hi = seed
        step = mp.mpf(step0)
        for _ in range(200):
            if sgn * (fn(hi) - target) >= 0:
                break
            hi *= step
            step = min(step * step, step_cap)
        else:
            raise RuntimeError("quantile bracket expansion failed (high side)")
    # Safeguarded false position in u = log(x), h = log(fn) - log(target)
    # coordinates. Bracket invariant: side(ulo) < 0 <= side(uhi) after sign
    # folding. h is used only to PLACE the interpolated step (it is the
    # near-linear coordinate in the tails); the plain sign decides which
    # bracket end moves, so an underflowed fn (exact 0, no log) degrades
    # a step to bisection instead of breaking anything.
    ltarget = mp.log(target)

    def _eval(u):
        v = fn(mp.exp(u))
        side = sgn * (v - target)
        smooth = sgn * (mp.log(v) - ltarget) if v > 0 else None
        return side, smooth

    ulo, uhi = mp.log(lo), mp.log(hi)
    _, flo = _eval(ulo)
    _, fhi = _eval(uhi)
    tol = mp.mpf("1e-45")  # in log-space = relative precision of x
    last_moved = 0  # +1 lo moved, -1 hi moved; two in a row forces bisection
    force_bisect = False
    # 600 steps: pure bisection of a log-width of 1e6 (x spanning e^+-5e5)
    # to 1e-45 takes 170; anti-stagnation forcing at most doubles that.
    # Until 2026-10-11 the loop stopped at 200 and returned the unconverged
    # midpoint silently.
    for _ in range(max_steps):
        width = uhi - ulo
        if width < tol:
            break
        if ulo + width / 2 in (ulo, uhi):
            break  # the bracket is two adjacent mpf values: converged at dps
        u_new = None
        if not force_bisect and flo is not None and fhi is not None and fhi != flo:
            u_try = ulo - flo * width / (fhi - flo)
            margin = width / 64
            if ulo + margin < u_try < uhi - margin:
                u_new = u_try
        if u_new is None:
            u_new = ulo + width / 2
            force_bisect = False
        side_new, f_new = _eval(u_new)
        if side_new < 0:
            ulo, flo = u_new, f_new
            force_bisect = last_moved == 1
            last_moved = 1
        else:
            uhi, fhi = u_new, f_new
            force_bisect = last_moved == -1
            last_moved = -1
    else:
        raise OracleNonConvergence(
            f"_tail_logspace_bisect: log-bracket width {mp.nstr(uhi - ulo, 3)} after {max_steps} steps "
            f"(target {mp.nstr(target, 6)})")
    return mp.exp((ulo + uhi) / 2)


try:
    from mpmath.libmp.libhyper import NoConvergence as _MPNoConvergence
except ImportError:  # pragma: no cover - future mpmath relocation guard

    class _MPNoConvergence(Exception):
        pass


GAMMA_QUAD_SHAPE = 2e4


def _gamma_cdf(alpha: "mp.mpf", beta: "mp.mpf", x: "mp.mpf") -> "mp.mpf":
    """Regularized lower incomplete gamma P(alpha, beta*x), hardened for
    large alpha the same way _betainc_reg is for large parameters:

    - shapes above GAMMA_QUAD_SHAPE: quadrature of the density
      (_gamma_pq_quad). mp.gammainc is slow there and raises NoConvergence
      near the mean; the switch-band sweep reaches alpha = 1e16.
    - far-lower-tail guard: P = y^alpha e^-y / (alpha Gamma(alpha)) * M with
      the series factor 1 <= M <= (alpha+1)/(alpha+1-y) for y < alpha+1, so
      when even the bound rounds to 0 in double the leading term is returned
      without the special-function call (it is the exact log-coordinate the
      quantile solver interpolates on). Until 2026-10-11 the guard was
      lead < -750 alone, sound only by margin: M exceeds e^5.6 (enough to lift
      a lead of -750 to a nonzero subnormal) only for alpha above ~1e8.
    - right of the mean (y > alpha) the LOWER-gamma series converges too
      slowly at large alpha and mpmath raises NoConvergence (observed at
      chi-squared k = 1e5, quantile solve); the UPPER-gamma continued
      fraction is the convergent representation there, so compute the
      complement. NoConvergence is mpmath's own exception class, NOT a
      ValueError -- catching it explicitly matters.
    - when mpmath raises anyway, the quadrature is the fallback, counted in
      FALLBACKS. (Until 2026-10-11 the fallback returned exp(lead), the
      leading term without its series factor, for any lead < -100.)
    """
    if x <= 0:
        return mp.mpf(0)
    y = beta * x
    if alpha > GAMMA_QUAD_SHAPE:
        return _gamma_pq_quad(alpha, y)[0]
    if y > alpha:
        # Right of the mean: complement via the upper-gamma continued
        # fraction. P >= ~0.5 here, so the tiny-value guard never applies.
        try:
            return 1 - mp.gammainc(alpha, y, mp.inf, regularized=True)
        except (ValueError, _MPNoConvergence):
            _note_fallback("gamma_cdf: gammainc raised, quadrature used")
            return _gamma_pq_quad(alpha, y)[0]
    # Left of the mean only: the y -> 0 leading term is a valid guard.
    lead, bound = _gamma_lower_lead(alpha, y)
    if bound < _LOG_HALF_DENORM - 1:
        return mp.exp(lead)
    try:
        return mp.gammainc(alpha, 0, y, regularized=True)
    except (ValueError, _MPNoConvergence):
        _note_fallback("gamma_cdf: gammainc raised, quadrature used")
        return _gamma_pq_quad(alpha, y)[0]


def _gamma_upper(alpha: "mp.mpf", y: "mp.mpf") -> "mp.mpf":
    """Regularized UPPER incomplete gamma Q(alpha, y) = 1 - P(alpha, y),
    hardened as _gamma_cdf's sibling and for the same reasons.

    It CANNOT be obtained as 1 - _gamma_cdf(alpha, 1, y): right of the mean
    _gamma_cdf itself returns 1 - Q, so the round trip is 1 - (1 - Q), and a
    Q of 1e-300 is gone at dps 50 the moment it is added to one. That
    cancellation is exactly what inverse_gamma's left tail is made of --
    F(x) = Q(alpha, beta/x) IS the small side there -- so the upper form has
    to be computed directly.

    - alpha > GAMMA_QUAD_SHAPE: quadrature, as in _gamma_cdf.
    - y <= alpha: P <= ~0.5, so 1 - P costs nothing and _gamma_cdf's own
      lower-tail guards carry over unchanged.
    - y > alpha: the upper continued fraction is the convergent
      representation (the same split _gamma_cdf makes, read from the other
      side). Far-tail guard: Q = y^(alpha-1) e^-y / Gamma(alpha) * U with
      U <= y / (y - alpha + 1) for y > alpha - 1; when the bound rounds to 0
      the leading term is returned (the quantile solvers' log-coordinate).
      If mpmath raises, the quadrature is the counted fallback.
    """
    if y <= 0:
        return mp.mpf(1)
    if alpha > GAMMA_QUAD_SHAPE:
        return _gamma_pq_quad(alpha, y)[1]
    if y <= alpha:
        return 1 - _gamma_cdf(alpha, mp.mpf(1), y)
    lead, bound = _gamma_upper_lead(alpha, y)
    if bound < _LOG_HALF_DENORM - 1:
        return mp.exp(lead)
    try:
        return mp.gammainc(alpha, y, mp.inf, regularized=True)
    except (ValueError, _MPNoConvergence):
        _note_fallback("gamma_upper: gammainc raised, quadrature used")
        return _gamma_pq_quad(alpha, y)[1]


def _norm_q(t: "mp.mpf") -> "mp.mpf":
    """Standard-normal upper tail Q(t) = Phi(-t) = erfc(t/sqrt(2))/2 -- full
    relative precision on its own small side. This is the erfc-domain tail
    piece TruncatedNormalDistribution::computeNormalization is built from
    (src/truncated_normal.cpp), replicated so the oracle and the library
    decompose the same window the same way. erfc(-inf) = 2 and erfc(+inf) = 0
    make it collapse correctly for infinite bounds."""
    return mp.erfc(t / SQRT2) / 2


def _phi_diff(lo: "mp.mpf", hi: "mp.mpf") -> "mp.mpf":
    """Phi(hi) - Phi(lo) by the regime split the library uses, never the
    cancelling difference of two CDFs.

    #57's own reference was first written as Phi(beta) - Phi(alpha) and had
    to be rewritten in survival form after the (37, 38)-sigma window ate it:
    both terms round to 1 and the difference is noise. Same three branches as
    computeNormalization:

      alpha >= 0    right tail, small - smaller:   Q(lo) - Q(hi)
      beta  <= 0    left tail, by reflection:      Q(-hi) - Q(-lo)
      otherwise     straddling, opposite signs:    (erf(hi) - erf(lo)) / 2

    Used for the normalization Z = _phi_diff(alpha, beta) AND for the CDF
    numerator _phi_diff(alpha, xi) -- they are the same function, so the
    numerator inherits the same cancellation-freedom rather than being an
    afterthought.
    """
    if lo >= 0:
        return _norm_q(lo) - _norm_q(hi)
    if hi <= 0:
        return _norm_q(-hi) - _norm_q(-lo)
    return (mp.erf(hi / SQRT2) - mp.erf(lo / SQRT2)) / 2


def _std_normal_quantile(pm: "mp.mpf") -> "mp.mpf":
    """Standard-normal quantile at the EXACT mpf p, via the survival form.

    NOT sqrt(2)*erfinv(2p - 1): at dps 50 the argument 2p - 1 rounds to
    exactly -1 for any p below ~1e-50 (verified: p = 1e-300 gives -1.0, and
    erfinv(-1) = -inf), which would turn every deep-tail quantile row into a
    spurious infinite-reference contract violation. Solving Q(t) = min(p, 1-p)
    on t > 0 keeps the target on its own small side at full precision, and
    1 - p is exact at dps 50 for a lifted double p >= 0.5 (Sterbenz).

    GaussianRef.quantile and LogNormalRef.quantile use this form since
    v2.4.2 (#157): their erfinv(2p - 1) form rounded to -inf for p <~ 1e-50,
    so libstats' matching -inf (#158) scored as correct. Before v2.4.2 they
    kept the erfinv form to freeze the pre-v2.4.0 baseline.
    """
    if pm == mp.mpf("0.5"):
        return mp.mpf(0)
    tail = pm if pm < mp.mpf("0.5") else 1 - pm
    t = _tail_logspace_bisect(_norm_q, tail, mp.mpf(1), increasing=False)
    return -t if pm < mp.mpf("0.5") else t


def _gamma_quantile(alpha: float, beta: float, p: float) -> "mp.mpf":
    a = mp.mpf(alpha)
    b = mp.mpf(beta)
    pm = mp.mpf(p)
    if pm <= 0:
        return mp.mpf(0)
    if pm >= 1:
        return mp.inf
    # Wilson-Hilferty approximation for the STANDARD (beta=1) gamma, used
    # only to seed the bracket. z from the survival-form normal quantile:
    # erfinv(2p - 1) is infinite for deep-tail p at dps 50.
    z = _std_normal_quantile(pm)
    guess = a * (1 - 1 / (9 * a) + z / (3 * mp.sqrt(a))) ** 3
    if guess <= 0 or not mp.isfinite(guess):
        guess = a
    if a > GAMMA_QUAD_SHAPE:
        if pm <= mp.mpf("0.5"):
            return _gamma_small_side_root(a, pm, True, guess) / b
        return _gamma_small_side_root(a, 1 - pm, False, guess) / b
    # Above alpha ~ 100 the seed is good to a few digits, so a near-1 first
    # bracket step saves evaluations (each a quadrature above
    # GAMMA_QUAD_SHAPE); squaring still recovers from a poor seed.
    step0 = 1 + 8 / mp.sqrt(a) if a > 100 else 4096
    return _tail_logspace_bisect(
        lambda xx: _gamma_cdf(a, b, xx), pm, guess / b, step0=step0)


def _gamma_small_side_root(a, target, lower, guess):
    """y with P(a, y) = target (lower) or Q(a, y) = target, for a above
    GAMMA_QUAD_SHAPE, where every evaluation is a quadrature: Newton on the
    small side from the Wilson-Hilferty seed. At these shapes the swept
    p >= 1e-300 puts y within a (1 +- 0.2), far inside the (a/4, 4a) bracket."""
    lga = mp.loggamma(a)
    log_dens = lambda yy: (a - 1) * mp.log(yy) - yy - lga
    return _newton_log_root(
        (lambda yy: _gamma_pq_quad(a, yy)[0]) if lower else (lambda yy: _gamma_pq_quad(a, yy)[1]),
        log_dens, target, a / 4, 4 * a, guess, increasing=lower)


def _gamma_quantile_upper_quad(a, q):
    """y with Q(a, y) = q, a above GAMMA_QUAD_SHAPE (inverse_gamma's left tail)."""
    z = -_std_normal_quantile(q)  # upper-tail normal quantile, survival form
    guess = a * (1 - 1 / (9 * a) + z / (3 * mp.sqrt(a))) ** 3
    return _gamma_small_side_root(a, q, False, guess)


# Discrete quantile searches give up past this count rather than loop.
_DISCRETE_SEARCH_LIMIT = 1 << 62


def _discrete_search_quantile(cdf_int, lo: int, hi, p: "mp.mpf", seed=None) -> int:
    """Smallest integer k in [lo, hi] with cdf_int(k) >= p (cdf_int monotone
    non-decreasing on integers; hi None for unbounded support).

    Gallops from `seed` (clamped into [lo, hi]) with doubling steps until the
    answer is bracketed, then bisects. With a Cornish-Fisher seed the answer
    is usually within a few counts, so a row costs ~5 CDF evaluations
    instead of the ~17 a bisection of [0, 2 lambda] took; at n = 1e4 each
    evaluation is an incomplete beta or gamma costing seconds. Until
    2026-10-11 the unbounded searches stopped doubling at 2^40 and returned
    whatever bracket they had; past _DISCRETE_SEARCH_LIMIT this raises.

    In exact arithmetic this is also the library's rule for the count families (Poisson,
    Binomial, NegativeBinomial, Geometric, Bernoulli): for p > 1/2 the library takes the
    smallest k with S(k) <= 1 - p, and 1 - p is exact for a double p in [1/2, 1], so
    F(k) >= p and S(k) <= 1 - p select the same k. The two differ only by the library's
    own rounding of S. Discrete keeps the library-CDF tie rule (DiscreteRef.quantile).
    """
    if p <= 0:
        return lo
    k = lo if seed is None else max(lo, int(seed))
    if hi is not None:
        k = min(k, hi)
    if cdf_int(k) >= p:
        # Answer in [lo, k]: gallop down.
        top, step = k, 1
        while True:
            cand = k - step
            if cand <= lo:
                bot = lo
                break
            if cdf_int(cand) < p:
                bot = cand + 1
                break
            top, step = cand, step * 2
        lo_, hi_ = bot, top
    else:
        # Answer in (k, hi]: gallop up.
        bot, step = k + 1, 1
        while True:
            cand = k + step
            if hi is not None and cand >= hi:
                top = hi
                break
            if cand > _DISCRETE_SEARCH_LIMIT:
                raise OracleNonConvergence(
                    f"discrete quantile search passed {_DISCRETE_SEARCH_LIMIT} (p = {p})")
            if cdf_int(cand) >= p:
                top = cand
                break
            bot, step = cand + 1, step * 2
        lo_, hi_ = bot, top
    while lo_ < hi_:
        mid = (lo_ + hi_) // 2
        if cdf_int(mid) >= p:
            hi_ = mid
        else:
            lo_ = mid + 1
    return lo_


def _cornish_fisher_seed(mean, var, skew, p: "mp.mpf") -> int:
    """Two-term Cornish-Fisher quantile seed for a discrete search."""
    z = _std_normal_quantile(mp.mpf(p))
    q = mp.mpf(mean) + mp.sqrt(var) * (z + mp.mpf(skew) * (z * z - 1) / 6)
    if not mp.isfinite(q) or q < 0:
        return 0
    return int(mp.floor(q))


def _poisson_cdf(lam: "mp.mpf", k: int) -> "mp.mpf":
    if k < 0:
        return mp.mpf(0)
    # Q(k + 1, lambda), through the hardened helper (quadrature above
    # GAMMA_QUAD_SHAPE, counted fallbacks) rather than raw mp.gammainc.
    return _gamma_upper(mp.mpf(k + 1), lam)


BETAINC_CF_MIN_SHAPE = 1000


def _betainc_reg(a: "mp.mpf", b: "mp.mpf", x: "mp.mpf") -> "mp.mpf":
    """Regularized incomplete beta I_x(a, b) with a far-tail asymptotic
    guard for huge parameters.

    mp.betainc routes through hyp2f1, and for a huge first parameter at an
    argument far from the transition region hypercomb FAILS TO CONVERGE --
    it escalates working precision (observed 189 -> 4577+ bits, with
    million-bit exponents in play) for minutes and then raises ValueError.
    That was the #46 oracle stall: student-t nu=1e6 probing t=100 dies
    inside betainc(5e5, 0.5, 0, 0.99).

    Guard: I_x(a, b) = x^a / (a B(a, b)) * 2F1(a, 1 - b; a + 1; x), and the
    2F1 factor is a * int_0^1 t^(a-1) (1 - x t)^(b-1) dt, at most 1 for
    b >= 1 and at most (1 - x)^(b-1) for b < 1. lead = a ln x - ln a -
    ln B(a, b); when lead plus that bound is below log(2^-1075) the value
    rounds to 0 in double, and exp(lead) is returned without calling betainc
    at all. exp(lead) stays monotone for the quantile solvers' bracketing
    and IS the correct leading log-coordinate their false-position phase
    interpolates on. (Until 2026-10-11 the guard was lead < -750 without the
    bound, and a betainc failure at lead < -100 returned exp(lead) -- the
    leading term without its O(1) factor -- silently. The continued
    fraction is now the counted fallback.)
    """
    if x <= 0:
        return mp.mpf(0)
    if x >= 1:
        return mp.mpf(1)
    lead, bound = _betainc_lead(a, b, x)
    if bound < _LOG_HALF_DENORM - 1:
        return mp.exp(lead)
    if min(a, b) >= BETAINC_CF_MIN_SHAPE:
        # BOTH parameters large: mp.betainc dies here too, even in the
        # central region where the value is ~0.5 and the lead guard
        # rightly does not fire (observed: betainc(7e5, 3e5, 0, 0.7)
        # hangs hyp2f1 for 300+ s; at min(a, b) ~ 5e3 a binomial n = 1e4
        # tail costs 5-80 s per call). The continued fraction converges in
        # ~O(sqrt(ab/(a+b))) cheap mpf iterations in exactly that regime,
        # and is self-checked against mp.betainc at min(a, b) ~ 1e3.
        return _betainc_cf(a, b, x)
    try:
        return mp.betainc(a, b, 0, x, regularized=True)
    except (ValueError, _MPNoConvergence):
        _note_fallback("betainc: mp.betainc raised, continued fraction used")
        return _betainc_cf(a, b, x)


def _betainc_cf(a: "mp.mpf", b: "mp.mpf", x: "mp.mpf") -> "mp.mpf":
    """I_x(a, b) by the standard continued-fraction expansion (modified
    Lentz iteration), written directly from the textbook formula:

        I_x(a,b) = x^a (1-x)^b / (a B(a,b)) * 1 / (1 + d1/(1 + d2/(1 + ...)))
        d_{2m}   = m (b - m) x / ((a + 2m - 1)(a + 2m))
        d_{2m+1} = -(a + m)(a + b + m) x / ((a + 2m)(a + 2m + 1))

    valid (rapidly convergent) for x < (a+1)/(a+b+2); the complement
    I_x(a,b) = 1 - I_{1-x}(b,a) covers the other side. Runs at elevated
    working precision so the ambient dps-50 result keeps full accuracy.
    Self-checked against mp.betainc on moderate parameters and against
    the exact symmetry point I_{1/2}(a,a) = 1/2 in run_self_checks.
    """
    if x > (a + 1) / (a + b + 2):
        return 1 - _betainc_cf(b, a, 1 - x)
    with mp.workprec(mp.mp.prec + 40):
        tiny = mp.mpf(2) ** (-2 * mp.mp.prec)
        eps = mp.mpf(2) ** (-mp.mp.prec + 5)
        # Modified Lentz for the CF part.
        c = mp.mpf(1)
        d = mp.mpf(1) - (a + b) * x / (a + 1)
        if abs(d) < tiny:
            d = tiny
        d = 1 / d
        h = d
        converged = False
        for m in range(1, 100000):
            m2 = 2 * m
            num = m * (b - m) * x / ((a + m2 - 1) * (a + m2))
            d = 1 + num * d
            if abs(d) < tiny:
                d = tiny
            c = 1 + num / c
            if abs(c) < tiny:
                c = tiny
            d = 1 / d
            h *= d * c
            num = -(a + m) * (a + b + m) * x / ((a + m2) * (a + m2 + 1))
            d = 1 + num * d
            if abs(d) < tiny:
                d = tiny
            c = 1 + num / c
            if abs(c) < tiny:
                c = tiny
            d = 1 / d
            delta = d * c
            h *= delta
            if abs(delta - 1) < eps:
                converged = True
                break
        if not converged:
            raise RuntimeError(
                f"incomplete beta CF failed to converge (a={a}, b={b}, x={x})")
        log_pre = (a * mp.log(x) + b * mp.log(1 - x) - mp.log(a)
                   - (mp.loggamma(a) + mp.loggamma(b) - mp.loggamma(a + b)))
        result = mp.exp(log_pre) * h
    return +result


def _binomial_cdf(n: int, p: "mp.mpf", k: int) -> "mp.mpf":
    if k < 0:
        return mp.mpf(0)
    if k >= n:
        return mp.mpf(1)
    return _betainc_reg(mp.mpf(n - k), mp.mpf(k + 1), 1 - p)


def _negbinom_cdf(r: "mp.mpf", p: "mp.mpf", k: int) -> "mp.mpf":
    if k < 0:
        return mp.mpf(0)
    return _betainc_reg(r, mp.mpf(k + 1), p)


STUDENT_T_QUAD_NU = 1e7


def _student_t_cdf(nu: "mp.mpf", t: "mp.mpf") -> "mp.mpf":
    if t == 0:
        return mp.mpf("0.5")
    if nu > STUDENT_T_QUAD_NU:
        # I_x(nu/2, 1/2) at x = nu/(nu + t^2) within 1e-7 of 1 for every
        # swept t: mp.betainc is slow and unverified there, so the density is
        # integrated instead (anchored at nu = 1e6 in the self-checks).
        return _student_t_cdf_quad(nu, t)
    xt = nu / (nu + t * t)
    ib = _betainc_reg(nu / 2, mp.mpf("0.5"), xt)
    return mp.mpf("0.5") * ib if t < 0 else 1 - mp.mpf("0.5") * ib


def _student_t_quantile(nu: "mp.mpf", p: "mp.mpf") -> "mp.mpf":
    if p == mp.mpf("0.5"):
        return mp.mpf(0)
    guess = SQRT2 * mp.erfinv(2 * p - 1)  # normal approx, seeds the bracket
    if not mp.isfinite(guess):
        # Deep tail: 2p-1 rounds to +-1 at dps 50 (p ~ 1e-300) and erfinv
        # blows up. Use the asymptotic normal quantile z^2 ~ 2L - ln(2*pi*
        # z^2), L = -ln(min(p, 1-p)), instead: without a finite seed the
        # bracket expansion starts at 1 and its doubling steps probe
        # extreme |t| where the large-nu incomplete beta costs MINUTES per
        # evaluation (the observed #46 stall at nu = 1e6). A ~1% seed is
        # plenty: expansion only needs to straddle the root, and the
        # false-position phase converges from there in a handful of steps.
        # (For small nu the true t-quantile is far beyond this normal-scale
        # seed -- t ~ 1e300 at nu=1 -- and the doubling expansion walks out
        # to it cheaply, small-nu evaluations being fast.)
        tail = p if p < mp.mpf("0.5") else 1 - p
        big_l = -mp.log(tail)
        z2 = 2 * big_l
        z2 = 2 * big_l - mp.log(2 * mp.pi * z2)
        guess = mp.sqrt(z2)
        if p < mp.mpf("0.5"):
            guess = -guess
    if nu > STUDENT_T_QUAD_NU:
        # Every evaluation is a quadrature here: Newton on the small side,
        # S(u) = F(-u) = 1 - F(u), decreasing in u = |t| (the swept p are at
        # least 0.25 from the median, so u > 0.6, inside (1e-6, 1e3)).
        n = nu
        C = mp.loggamma((n + 1) / 2) - mp.loggamma(n / 2) - mp.log(n * mp.pi) / 2
        tail = p if p < mp.mpf("0.5") else 1 - p
        z = abs(guess)
        u = _newton_log_root(lambda uu: _student_t_cdf_quad(n, -uu),
                             lambda uu: C - (n + 1) / 2 * mp.log1p(uu * uu / n),
                             tail, mp.mpf("1e-6"), mp.mpf(1000),
                             z + (z**3 + z) / (4 * n), increasing=False)
        return -u if p < mp.mpf("0.5") else u
    step0 = 4096
    if nu > 1000:
        # Cornish-Fisher: t = z + (z^3 + z) / (4 nu) + O(nu^-2). Accurate to
        # a few digits from nu = 1e3, so the bracket starts tight (each
        # evaluation is a quadrature above STUDENT_T_QUAD_NU).
        guess = guess + (guess**3 + guess) / (4 * nu)
        step0 = 1 + mp.mpf(1) / mp.sqrt(nu)
    if p > mp.mpf("0.5"):
        # t* > 0; F is increasing from 0.5 to 1 on (0, inf).
        return _tail_logspace_bisect(
            lambda t: _student_t_cdf(nu, t), p, guess, step0=step0)
    # p < 0.5: solve on the negative axis directly through the
    # cancellation-free incomplete-beta branch (F(-u) = 0.5*I(...) is
    # DECREASING in u on (0, inf), range (0, 0.5)). Never via the
    # symmetry Q(p) = -Q(1-p): for p ~ 1e-300, 1-p rounds to exactly 1
    # at dps 50 and the reflected problem degenerates.
    return -_tail_logspace_bisect(
        lambda u: _student_t_cdf(nu, -u), p, -guess, increasing=False, step0=step0)


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
        return mp.mpf(mean) + mp.mpf(sigma) * _std_normal_quantile(mp.mpf(p))


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
        z = _std_normal_quantile(mp.mpf(p))
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
        # expm1/log1p in every lower-tail reference (Exponential, Geometric, Pareto, Rayleigh,
        # Weibull): at dps 50, 1 - e^-t keeps no digit of a t below ~1e-50, and the library's
        # log1p quantiles now put sweep points there.
        return -mp.expm1(-l * xm) if xm >= 0 else mp.mpf(0)

    @staticmethod
    def quantile(lam, _p2, p):
        return -mp.log1p(-mp.mpf(p)) / mp.mpf(lam)


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
        if xm < 0:
            return mp.mpf(0)
        if xm == 0:
            # The limit at the support boundary (#161): +inf for alpha < 1, beta for
            # alpha = 1, 0 for alpha > 1.
            return mp.inf if a < 1 else (b if a == 1 else mp.mpf(0))
        return b**a / mp.gamma(a) * xm ** (a - 1) * mp.e ** (-b * xm)

    @staticmethod
    def logpdf(alpha, beta, x):
        a, b, xm = mp.mpf(alpha), mp.mpf(beta), mp.mpf(x)
        if xm < 0:
            return -mp.inf
        if xm == 0:
            return mp.inf if a < 1 else (mp.log(b) if a == 1 else -mp.inf)
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
        k = min(max(a - 1 + int(mp.ceil(pm * n)), a), b)
        # The contract is the smallest k whose *library* CDF reaches p (Decided; #116, #170,
        # #204), and the library's F(k) is the double (k - a + 1) / n. At a tie, where p equals
        # that double but not the exact (k - a + 1) / n, the two rules differ by one count.
        pf = float(p)
        while k > a and float(k - a) / n >= pf:
            k -= 1
        while k < b and float(k - a + 1) / n < pf:
            k += 1
        return mp.mpf(k)


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
        # Tail form: F = atan(1/|z|)/pi on the left, 1 - that on the right.
        # The former 0.5 + atan(z)/pi cancels in the left tail: at dps 50 it
        # keeps no digit of an F below ~1e-50, and the swept left tail reaches
        # F ~ 1e-300 (scored as 1.0 relative error on correct library rows).
        x0, g, xm = mp.mpf(x0), mp.mpf(gamma), mp.mpf(x)
        z = (xm - x0) / g
        if z == 0:
            return mp.mpf("0.5")
        tail = mp.atan(1 / abs(z)) / mp.pi
        return tail if z < 0 else 1 - tail

    @staticmethod
    def quantile(x0, gamma, p):
        # cot(pi*q) on the smaller tail: at dps 50, p - 1/2 loses a p below ~1e-50 (1e-300 gave
        # tan(-pi/2), +2e51, for the true -3.2e299).
        pm = mp.mpf(p)
        if pm == mp.mpf("0.5"):
            return mp.mpf(x0)  # exactly; cot(pi/2) at dps 50 is -5e-52, not 0
        if pm < mp.mpf("0.5"):
            return mp.mpf(x0) - mp.mpf(gamma) * mp.cot(mp.pi * pm)
        return mp.mpf(x0) + mp.mpf(gamma) * mp.cot(mp.pi * (1 - pm))


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
        q = 1 - pm
        seed = _cornish_fisher_seed(n * pm, n * pm * q, (q - pm) / mp.sqrt(n * pm * q), pr) \
            if 0 < pm < 1 else 0
        return mp.mpf(_discrete_search_quantile(lambda k: _binomial_cdf(n, pm, k), 0, n, pr, seed))


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
        # Seeded gallop, unbounded above (no fixed support ceiling for the
        # failure count). Failures: mean r q / p, variance r q / p^2, skew
        # (2 - p) / sqrt(r q).
        q = 1 - pm
        seed = _cornish_fisher_seed(rm * q / pm, rm * q / pm**2, (2 - pm) / mp.sqrt(rm * q), pr) \
            if pm < 1 else 0
        return mp.mpf(_discrete_search_quantile(lambda k: _negbinom_cdf(rm, pm, k), 0, None, pr,
                                                seed))


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
        if k < 0:
            return -mp.inf
        if k == 0:
            # k * log1p(-p) is exactly 0 here; spelling it out avoids the
            # 0 * (-inf) = NaN artifact at p == 1 (pmf(0) = p exactly).
            return mp.log(pm)
        return mp.log(pm) + k * mp.log1p(-pm)

    @staticmethod
    def cdf(p, _p2, x):
        pm = mp.mpf(p)
        k = int(math.floor(x))
        if k < 0:
            return mp.mpf(0)
        return -mp.expm1((k + 1) * mp.log1p(-pm))

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
    def _edge_pdf(edge_shape, other_shape):
        # Density AT an endpoint: the endpoint's shape exponent governs.
        # shape < 1 is an integrable singularity (+inf is the limit, and
        # what the library returns); shape == 1 leaves the finite value
        # 1/B(1, other) = other; shape > 1 pins the density to 0.
        if edge_shape < 1:
            return mp.inf
        if edge_shape == 1:
            return other_shape
        return mp.mpf(0)

    @staticmethod
    def pdf(alpha, beta, x):
        a, b, xm = mp.mpf(alpha), mp.mpf(beta), mp.mpf(x)
        if xm < 0 or xm > 1:
            return mp.mpf(0)
        if xm == 0:
            return BetaRef._edge_pdf(a, b)
        if xm == 1:
            return BetaRef._edge_pdf(b, a)
        return xm ** (a - 1) * (1 - xm) ** (b - 1) / mp.beta(a, b)

    @staticmethod
    def logpdf(alpha, beta, x):
        a, b, xm = mp.mpf(alpha), mp.mpf(beta), mp.mpf(x)
        if xm < 0 or xm > 1:
            return -mp.inf
        if xm == 0 or xm == 1:
            edge = BetaRef._edge_pdf(a if xm == 0 else b, b if xm == 0 else a)
            return mp.log(edge) if edge > 0 else -mp.inf
        return (a - 1) * mp.log(xm) + (b - 1) * mp.log1p(-xm) - mp.log(mp.beta(a, b))

    @staticmethod
    def cdf(alpha, beta, x):
        a, b, xm = mp.mpf(alpha), mp.mpf(beta), mp.mpf(x)
        if xm <= 0:
            return mp.mpf(0)
        if xm >= 1:
            return mp.mpf(1)
        return _betainc_reg(a, b, xm)

    @staticmethod
    def quantile(alpha, beta, p):
        a, b, pm = mp.mpf(alpha), mp.mpf(beta), mp.mpf(p)
        if pm <= 0:
            return mp.mpf(0)
        if pm >= 1:
            return mp.mpf(1)

        def q_lower(aa, bb, ppm):
            # Lower-half solve on (0, 1]; hi_cap=1 is always a valid upper
            # bracket (I_1 = 1 >= ppm), the low side log-expands toward 0.
            return _tail_logspace_bisect(
                lambda xx: _betainc_reg(aa, bb, xx),
                ppm, aa / (aa + bb), hi_cap=1)

        if pm > mp.mpf("0.5"):
            # Reflect: Q(a,b,p) = 1 - Q(b,a,1-p). Here 1-pm IS exact at
            # dps 50 (pm is a lifted double in (0.5, 1), so 1-pm needs
            # < 53 bits), unlike the p < 0.5 deep-tail direction -- and it
            # moves the solve to the log-space-friendly lower tail.
            return 1 - q_lower(b, a, 1 - pm)
        return q_lower(a, b, pm)


@_reg("chi_squared")
class ChiSquaredRef(Ref):
    @staticmethod
    def pdf(k, _p2, x):
        kk, xm = mp.mpf(k), mp.mpf(x)
        if xm < 0:
            return mp.mpf(0)
        if xm == 0:
            # Gamma(k/2, 1/2) at the boundary (#161): +inf for k < 2, 1/2 for k = 2, else 0.
            return mp.inf if kk < 2 else (mp.mpf("0.5") if kk == 2 else mp.mpf(0))
        return xm ** (kk / 2 - 1) * mp.e ** (-xm / 2) / (2 ** (kk / 2) * mp.gamma(kk / 2))

    @staticmethod
    def logpdf(k, _p2, x):
        kk, xm = mp.mpf(k), mp.mpf(x)
        if xm < 0:
            return -mp.inf
        if xm == 0:
            return mp.inf if kk < 2 else (-mp.log(2) if kk == 2 else -mp.inf)
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
        return -mp.expm1(-a * mp.log1p((xm - xm0) / xm0))

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
        return -mp.expm1(-xm * xm / (2 * s * s))

    @staticmethod
    def quantile(sigma, _p2, p):
        s, pm = mp.mpf(sigma), mp.mpf(p)
        return s * mp.sqrt(-2 * mp.log1p(-pm))


@_reg("weibull")
class WeibullRef(Ref):
    @staticmethod
    def pdf(shape, scale, x):
        k, lam, xm = mp.mpf(shape), mp.mpf(scale), mp.mpf(x)
        if xm < 0:
            return mp.mpf(0)
        if xm == 0:
            # The same boundary rule as logpdf below (#161), stated rather than left to
            # mpmath's 0 ** (k - 1).
            return mp.inf if k < 1 else (1 / lam if k == 1 else mp.mpf(0))
        return (k / lam) * (xm / lam) ** (k - 1) * mp.e ** (-((xm / lam) ** k))

    @staticmethod
    def logpdf(shape, scale, x):
        k, lam, xm = mp.mpf(shape), mp.mpf(scale), mp.mpf(x)
        if xm < 0:
            return -mp.inf
        if xm == 0:
            # Density at 0: +inf for k < 1 (integrable singularity, what
            # the library returns), k/lam = 1/lam for k == 1, 0 for k > 1.
            if k < 1:
                return mp.inf
            if k == 1:
                return -mp.log(lam)
            return -mp.inf
        return mp.log(k) - mp.log(lam) + (k - 1) * (mp.log(xm) - mp.log(lam)) - (xm / lam) ** k

    @staticmethod
    def cdf(shape, scale, x):
        k, lam, xm = mp.mpf(shape), mp.mpf(scale), mp.mpf(x)
        if xm < 0:
            return mp.mpf(0)
        return -mp.expm1(-((xm / lam) ** k))

    @staticmethod
    def quantile(shape, scale, p):
        k, lam, pm = mp.mpf(shape), mp.mpf(scale), mp.mpf(p)
        # log1p: at dps 50, 1 - p rounds a p below ~1e-50 to 1 and the reference to 0.
        return lam * (-mp.log1p(-pm)) ** (1 / k)


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
        seed = _cornish_fisher_seed(l, l, 1 / mp.sqrt(l), pr)
        return mp.mpf(_discrete_search_quantile(lambda k: _poisson_cdf(l, k), 0, None, pr, seed))


# ---------------------------------------------------------------------------
# v2.4.0 distributions (#54/#55/#56/#57). Appended, never interleaved: the
# pre-v2.4.0 references above are frozen, because the 63-violation comparison
# baseline they produced is what the v2.5.0 corvus before/after is measured
# against.
#
# Every reference below routes its special functions through the hardened
# helpers at the top of this file (_betainc_reg / _betainc_cf, _gamma_cdf /
# _gamma_upper, _tail_logspace_bisect, _phi_diff). None of them calls
# mp.betainc or mp.gammainc directly: the #46 history is that raw mpmath
# hangs or raises for min(a, b) >~ 5e3 and for far-tail arguments, and
# fisher_f (d1 = d2 = 1e4 puts BOTH incomplete-beta parameters at 5000) and
# inverse_gamma (its CDF is an upper incomplete gamma read from the small
# side) are precisely the two that would have rediscovered it.
# ---------------------------------------------------------------------------


@_reg("logistic")
class LogisticRef(Ref):
    @staticmethod
    def _z(mu, s, x):
        return (mp.mpf(x) - mp.mpf(mu)) / mp.mpf(s)

    @staticmethod
    def pdf(mu, s, x):
        # Evaluated at |z|: the density is symmetric, and the folded form
        # keeps exp() on its decaying side for either tail.
        a = abs(LogisticRef._z(mu, s, x))
        e = mp.exp(-a)
        return e / (mp.mpf(s) * (1 + e) ** 2)

    @staticmethod
    def logpdf(mu, s, x):
        a = abs(LogisticRef._z(mu, s, x))
        return -a - 2 * mp.log1p(mp.exp(-a)) - mp.log(mp.mpf(s))

    @staticmethod
    def cdf(mu, s, x):
        z = LogisticRef._z(mu, s, x)
        if z < 0:
            e = mp.exp(z)
            return e / (1 + e)
        return 1 / (1 + mp.exp(-z))

    @staticmethod
    def quantile(mu, s, p):
        # log p - log(1 - p) at the EXACT double of p (#56): 1 - pm is exact
        # at dps 50 for a lifted double, so no complement is invented here.
        pm = mp.mpf(p)
        return mp.mpf(mu) + mp.mpf(s) * (mp.log(pm) - mp.log(1 - pm))


@_reg("gumbel")
class GumbelRef(Ref):
    @staticmethod
    def _z(mu, beta, x):
        return (mp.mpf(x) - mp.mpf(mu)) / mp.mpf(beta)

    @staticmethod
    def pdf(mu, beta, x):
        z = GumbelRef._z(mu, beta, x)
        return mp.exp(-z - mp.exp(-z)) / mp.mpf(beta)

    @staticmethod
    def logpdf(mu, beta, x):
        z = GumbelRef._z(mu, beta, x)
        return -mp.log(mp.mpf(beta)) - z - mp.exp(-z)

    @staticmethod
    def cdf(mu, beta, x):
        return mp.exp(-mp.exp(-GumbelRef._z(mu, beta, x)))

    @staticmethod
    def quantile(mu, beta, p):
        return mp.mpf(mu) - mp.mpf(beta) * mp.log(-mp.log(mp.mpf(p)))


@_reg("bernoulli")
class BernoulliRef(Ref):
    @staticmethod
    def pdf(p, _p2, x):
        pm = mp.mpf(p)
        k = int(round(x))
        if k == 1:
            return pm
        if k == 0:
            return 1 - pm
        return mp.mpf(0)

    @staticmethod
    def logpdf(p, _p2, x):
        v = BernoulliRef.pdf(p, _p2, x)
        return mp.log(v) if v > 0 else -mp.inf

    @staticmethod
    def cdf(p, _p2, x):
        k = int(math.floor(x))
        if k < 0:
            return mp.mpf(0)
        if k >= 1:
            return mp.mpf(1)
        return 1 - mp.mpf(p)

    @staticmethod
    def quantile(p, _p2, prob):
        return mp.mpf(
            _discrete_search_quantile(lambda k: BernoulliRef.cdf(p, _p2, k), 0, 1, mp.mpf(prob))
        )


@_reg("erlang")
class ErlangRef(Ref):
    """Erlang(k, lambda) = Gamma(alpha=k, beta=lambda) exactly, in libstats'
    shape-RATE Gamma parameterization (erlang.h). The CDF and quantile
    therefore reuse GammaRef's hardened machinery unchanged rather than
    restating it -- an independent second incomplete-gamma implementation
    here would be a second thing to get wrong, not a stronger check."""

    @staticmethod
    def _k(k):
        return mp.mpf(int(round(k)))

    @staticmethod
    def logpdf(k, lam, x):
        kk, l, xm = ErlangRef._k(k), mp.mpf(lam), mp.mpf(x)
        if xm < 0:
            return -mp.inf
        if xm == 0:
            # k = 1 is the exponential, whose density at 0 is lambda; every
            # k > 1 pins it to 0. (k < 1 is impossible: the factory rejects.)
            return mp.log(l) if kk == 1 else -mp.inf
        return kk * mp.log(l) - mp.loggamma(kk) + (kk - 1) * mp.log(xm) - l * xm

    @staticmethod
    def pdf(k, lam, x):
        xm = mp.mpf(x)
        if xm < 0:
            return mp.mpf(0)
        if xm == 0:
            return mp.mpf(lam) if ErlangRef._k(k) == 1 else mp.mpf(0)
        # Exponentiated log form, not b**a / gamma(a) * ...: at k = 10000 with
        # lambda = 1e-3 the separate factors are 1e-30000 and 1e35659, which
        # mpmath carries but which say nothing the log form does not.
        return mp.exp(ErlangRef.logpdf(k, lam, x))

    @staticmethod
    def cdf(k, lam, x):
        return _gamma_cdf(ErlangRef._k(k), mp.mpf(lam), mp.mpf(x))

    @staticmethod
    def quantile(k, lam, p):
        return _gamma_quantile(float(int(round(k))), float(lam), p)


@_reg("fisher_f")
class FisherFRef(Ref):
    """F(d1, d2): CDF is the regularized incomplete beta I_y(d1/2, d2/2) at
    y = d1 x / (d1 x + d2) (fisher_f.h).

    Both the CDF and the quantile are STEERED rather than computed and then
    complemented. y and 1 - y are each formed directly from d1*x and d2 over
    the common denominator, so neither is ever a rounded complement of the
    other, and the quantile solves for whichever of the two is the small one.
    """

    @staticmethod
    def _ab(d1, d2):
        return mp.mpf(d1) / 2, mp.mpf(d2) / 2

    @staticmethod
    def _log_lead_at_zero(d1, d2):
        """log of lim_{x->0} f(x)/x^(a-1) = a*log(d1) - a*log(d2) - log B(a,b)."""
        a, b = FisherFRef._ab(d1, d2)
        logB = mp.loggamma(a) + mp.loggamma(b) - mp.loggamma(a + b)
        return a * mp.log(mp.mpf(d1)) - a * mp.log(mp.mpf(d2)) - logB

    @staticmethod
    def logpdf(d1, d2, x):
        dd1, dd2, xm = mp.mpf(d1), mp.mpf(d2), mp.mpf(x)
        if xm < 0:
            return -mp.inf
        a, b = FisherFRef._ab(d1, d2)
        if xm == 0:
            # f ~ x^(a-1) * lead. a < 1 is an integrable singularity (+inf),
            # a == 1 leaves the finite lead, a > 1 pins the density to 0.
            # d1 = 0.01 (a = 0.005) reaches this: its deep-left quantile
            # underflows to exactly 0.0 in double, so x = 0 IS on the grid.
            if a < 1:
                return mp.inf
            if a == 1:
                return FisherFRef._log_lead_at_zero(d1, d2)
            return -mp.inf
        logB = mp.loggamma(a) + mp.loggamma(b) - mp.loggamma(a + b)
        return (
            a * mp.log(dd1 * xm)
            + b * mp.log(dd2)
            - mp.log(xm)
            - (a + b) * mp.log(dd1 * xm + dd2)
            - logB
        )

    @staticmethod
    def pdf(d1, d2, x):
        lp = FisherFRef.logpdf(d1, d2, x)
        if lp == mp.inf:
            return mp.inf
        if lp == -mp.inf:
            return mp.mpf(0)
        return mp.exp(lp)

    @staticmethod
    def cdf(d1, d2, x):
        dd1, dd2, xm = mp.mpf(d1), mp.mpf(d2), mp.mpf(x)
        if xm <= 0:
            return mp.mpf(0)
        a, b = FisherFRef._ab(d1, d2)
        num = dd1 * xm
        den = num + dd2
        y, yc = num / den, dd2 / den
        # Same convergence split _betainc_cf makes internally, applied one
        # level up so the complement argument is the directly-computed yc
        # rather than a rounded 1 - y.
        if y <= (a + 1) / (a + b + 2):
            return _betainc_reg(a, b, y)
        return 1 - _betainc_reg(b, a, yc)

    @staticmethod
    def quantile(d1, d2, p):
        dd1, dd2, pm = mp.mpf(d1), mp.mpf(d2), mp.mpf(p)
        if pm <= 0:
            return mp.mpf(0)
        if pm >= 1:
            return mp.inf
        a, b = FisherFRef._ab(d1, d2)

        def q_lower(aa, bb, ppm):
            # BetaRef's shape: hi_cap=1 keeps the bracket finite, the low
            # side log-expands toward 0.
            return _tail_logspace_bisect(
                lambda yy: _betainc_reg(aa, bb, yy), ppm, aa / (aa + bb), hi_cap=1
            )

        # Solve for whichever of y, yc = 1 - y is below 1/2 at the root, decided
        # by comparing p with the CDF at y = 1/2 (x = d2/d1). Until 2026-10-11
        # the branch followed p > 1/2, so a root with y near 1 was solved in y
        # and mapped through y/(1 - y) with 1 - y rounded away: F(2, 0.002),
        # whose median has 1 - y = 9.3e-302, came out 2.1e42 for 1.07e298.
        if pm > _betainc_reg(a, b, mp.mpf("0.5")):
            # yc on ITS own lower tail (I_yc(b, a) = 1 - p), mapped back as
            # x = (d2/d1) (1 - yc)/yc. 1 - pm is exact at dps 50 for a lifted
            # double pm >= 2^-100 (its last bit is then above 2^-160).
            if pm < mp.mpf(2) ** -100:
                raise OracleNonConvergence(
                    f"fisher_f quantile: 1 - p inexact at dps 50 (p = {mp.nstr(pm, 6)})")
            yc = q_lower(b, a, 1 - pm)
            return (dd2 / dd1) * (1 - yc) / yc
        y = q_lower(a, b, pm)
        return (dd2 / dd1) * y / (1 - y)


@_reg("inverse_gamma")
class InverseGammaRef(Ref):
    """InvGamma(alpha, beta) with beta a SCALE (inverse_gamma.h, scipy's
    invgamma(a, scale=beta)). CDF is the UPPER regularized incomplete gamma
    Q(alpha, beta/x) -- read from the small side in the left tail, which is
    why _gamma_upper exists rather than 1 - _gamma_cdf(...)."""

    @staticmethod
    def logpdf(alpha, beta, x):
        a, b, xm = mp.mpf(alpha), mp.mpf(beta), mp.mpf(x)
        if xm <= 0:
            # x = 0 is reachable (alpha = 0.01's deep-left quantile underflows
            # to exactly 0.0 in double). exp(-beta/x) -> 0 beats every power.
            return -mp.inf
        return a * mp.log(b) - mp.loggamma(a) - (a + 1) * mp.log(xm) - b / xm

    @staticmethod
    def pdf(alpha, beta, x):
        lp = InverseGammaRef.logpdf(alpha, beta, x)
        return mp.mpf(0) if lp == -mp.inf else mp.exp(lp)

    @staticmethod
    def cdf(alpha, beta, x):
        a, b, xm = mp.mpf(alpha), mp.mpf(beta), mp.mpf(x)
        if xm <= 0:
            return mp.mpf(0)
        return _gamma_upper(a, b / xm)

    @staticmethod
    def quantile(alpha, beta, p):
        a, b, pm = mp.mpf(alpha), mp.mpf(beta), mp.mpf(p)
        if pm <= 0:
            return mp.mpf(0)
        if pm >= 1:
            return mp.inf
        # Substitute u = beta/x, so F(x) = Q(alpha, u) is DECREASING in u:
        # solve in u, invert at the end. Each branch is chosen so the solve
        # target is the small, cancellation-free side.
        if pm > mp.mpf("0.5"):
            # P(alpha, u) = 1 - p, exact at dps 50 (and in double: Sterbenz
            # makes 1.0 - p exact for a double p >= 0.5), so this hands the
            # existing hardened gamma-quantile solver an exact target.
            u = _gamma_quantile(float(alpha), 1.0, 1.0 - float(p))
        elif a > GAMMA_QUAD_SHAPE:
            u = _gamma_quantile_upper_quad(a, pm)
        else:
            u = _tail_logspace_bisect(lambda uu: _gamma_upper(a, uu), pm, a, increasing=False)
        return mp.inf if u <= 0 else b / u


@_reg("half_normal")
class HalfNormalRef(Ref):
    @staticmethod
    def pdf(sigma, _p2, x):
        s, xm = mp.mpf(sigma), mp.mpf(x)
        if xm < 0:
            return mp.mpf(0)
        return mp.sqrt(2 / mp.pi) / s * mp.exp(-xm * xm / (2 * s * s))

    @staticmethod
    def logpdf(sigma, _p2, x):
        s, xm = mp.mpf(sigma), mp.mpf(x)
        if xm < 0:
            return -mp.inf
        return mp.log(2 / mp.pi) / 2 - mp.log(s) - xm * xm / (2 * s * s)

    @staticmethod
    def cdf(sigma, _p2, x):
        s, xm = mp.mpf(sigma), mp.mpf(x)
        if xm <= 0:
            return mp.mpf(0)
        return mp.erf(xm / (s * SQRT2))

    @staticmethod
    def quantile(sigma, _p2, p):
        pm = mp.mpf(p)
        if pm <= 0:
            return mp.mpf(0)
        if pm >= 1:
            return mp.inf
        # erfinv's argument is p ITSELF, not 2p - 1, so the half-normal
        # quantile has none of the deep-tail cancellation that collapses the
        # symmetric-Gaussian form (verified: mp.erfinv(1e-300) is exact to
        # 50 digits, while mp.erfinv(2*mpf(1e-300) - 1) is -inf). This is the
        # oracle side being well-conditioned; the library's own erf_inv is
        # the one #136 files against, and these rows are where it shows.
        return mp.mpf(sigma) * SQRT2 * mp.erfinv(pm)


@_reg("truncated_normal")
class TruncatedNormalRef(Ref):
    """TruncatedNormal(mean, sd, a, b). p1/p2 carry (mean, sd) and the window
    comes from _TN_WINDOWS -- see the input contract in this file's header.

    Built entirely from erfc-domain tail pieces via _phi_diff, per #57: a
    Phi-difference reference cancels to noise on a same-tail window, which
    the (10, 12) instance (standardized alpha=5, beta=6, Z ~ 2.86e-7) is
    there to exercise.
    """

    @staticmethod
    def _std(mean, sd, x):
        a, b = _tn_window(mean, sd)
        m, s = mp.mpf(mean), mp.mpf(sd)
        return (mp.mpf(a) - m) / s, (mp.mpf(b) - m) / s, (mp.mpf(x) - m) / s, a, b

    @staticmethod
    def logpdf(mean, sd, x):
        alpha, beta, xi, a, b = TruncatedNormalRef._std(mean, sd, x)
        if x < a or x > b:
            return -mp.inf
        z = _phi_diff(alpha, beta)
        return -xi * xi / 2 - LOG_2PI / 2 - mp.log(mp.mpf(sd)) - mp.log(z)

    @staticmethod
    def pdf(mean, sd, x):
        lp = TruncatedNormalRef.logpdf(mean, sd, x)
        return mp.mpf(0) if lp == -mp.inf else mp.exp(lp)

    @staticmethod
    def cdf(mean, sd, x):
        alpha, beta, xi, a, b = TruncatedNormalRef._std(mean, sd, x)
        if x <= a:
            return mp.mpf(0)
        if x >= b:
            return mp.mpf(1)
        # Numerator and normalization are the SAME cancellation-free
        # function, evaluated on (alpha, xi) and (alpha, beta).
        return _phi_diff(alpha, xi) / _phi_diff(alpha, beta)

    @staticmethod
    def quantile(mean, sd, p):
        a, b = _tn_window(mean, sd)
        m, s, pm = mp.mpf(mean), mp.mpf(sd), mp.mpf(p)
        if pm <= 0:
            return mp.mpf(a)
        if pm >= 1:
            return mp.mpf(b)
        alpha, beta = (mp.mpf(a) - m) / s, (mp.mpf(b) - m) / s
        if not (mp.isfinite(alpha) or mp.isfinite(beta)):
            # Doubly infinite: exactly the plain Gaussian. Uses the survival
            # -form solver, not sqrt(2)*erfinv(2p - 1), which would return
            # -inf for every p below ~1e-50 (see _std_normal_quantile).
            return m + s * _std_normal_quantile(pm)
        # Bisect on the small side: the lower mass _phi_diff(alpha, xi) = p Z
        # for p <= 1/2, the upper mass _phi_diff(xi, beta) = (1 - p) Z above
        # it (1 - p is exact at dps 50 for a lifted double p >= 1/2). Both are
        # cancellation-free and monotone in xi, so bisection cannot stall;
        # unlike a closed-form Phi^-1(Phi(alpha) + p Z) it never forms the
        # sum #57 showed cancels. An infinite bound is replaced by a finite
        # one 40 sigma past the finite bound or the origin, far enough that
        # the mass beyond it (Q(40) ~ 4e-350, relative to Z) is below every
        # swept p; 300 halvings of the resulting width land ~1e-88, far
        # inside dps 50. (Until 2026-10-11 a singly-infinite window raised.)
        z = _phi_diff(alpha, beta)
        lower = pm <= mp.mpf("0.5")
        target = pm * z if lower else (1 - pm) * z
        lo = alpha if mp.isfinite(alpha) else min(beta, mp.mpf(0)) - 40 - mp.sqrt(-2 * mp.log(pm))
        hi = beta if mp.isfinite(beta) else max(alpha, mp.mpf(0)) + 40 + mp.sqrt(-2 * mp.log(1 - pm))
        for _ in range(300):
            mid = (lo + hi) / 2
            d = _phi_diff(alpha, mid) if lower else _phi_diff(mid, beta)
            if d == target:
                # Exact hit: return the root itself rather than a bracket
                # midpoint 1e-51 away from it. This fires on the symmetric
                # median -- TN(0, 1, -2, 2) at p = 0.5 has the root exactly 0,
                # and _phi_diff(-2, 0) equals 0.5*Z exactly at mpf precision.
                # Without it the oracle reports a ~1e-51 reference against the
                # library's exactly-correct 0.0, which scores as relative
                # error 1.0 and reads as a finding when it is the oracle's own
                # bracket residue.
                lo = hi = mid
                break
            if (d < target) == lower:
                lo = mid
            else:
                hi = mid
        return m + s * (lo + hi) / 2


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
        if os.environ.get("ORACLE_CHECK_TIMES"):
            print(f"  [check] {name}", file=sys.stderr, flush=True)
        assert cond, f"SELF-CHECK FAILED: {name} {detail}"
        results.append((name, True, detail))

    tol = mp.mpf("1e-30")

    # --- incomplete-beta continued fraction (large-parameter path) ---
    # The CF replaces mp.betainc wherever min(a,b) >= BETAINC_CF_MIN_SHAPE
    # (see _betainc_reg); anchor it against mp.betainc where mpmath is
    # healthy -- including both tails at the switch's own scale -- and
    # against the exact symmetry point I_{1/2}(a,a) = 1/2 in the
    # large-parameter regime mpmath cannot reach.
    for _a, _b, _x in ((2.5, 7.0, 0.2), (30.0, 4.0, 0.9), (100.0, 250.0, 0.31),
                       (1200.0, 1000.0, 0.47), (1200.0, 1000.0, 0.6),
                       (1000.0, 1200.0, 0.5)):
        _ref = mp.betainc(mp.mpf(_a), mp.mpf(_b), 0, mp.mpf(_x), regularized=True)
        _cf = _betainc_cf(mp.mpf(_a), mp.mpf(_b), mp.mpf(_x))
        check(
            f"betainc_cf.vs_mpmath({_a},{_b},{_x})",
            abs(_cf - _ref) <= mp.mpf("1e-45") * _ref,
        )
    check(
        "betainc_cf.symmetry_point_large",
        abs(_betainc_cf(mp.mpf(10000), mp.mpf(10000), mp.mpf("0.5")) - mp.mpf("0.5"))
        <= mp.mpf("1e-45"),
    )

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

    # -----------------------------------------------------------------------
    # v2.4.0 helpers and distributions. Two independent checks per
    # distribution, same doctrine as above; the helper checks come first
    # because every reference below leans on them.
    # -----------------------------------------------------------------------

    # --- _gamma_upper (the CDF of inverse_gamma, read from the small side) ---
    check(
        "gamma_upper.complements_gamma_cdf",
        abs(_gamma_upper(mp.mpf(3), mp.mpf(2)) - (1 - _gamma_cdf(mp.mpf(3), mp.mpf(1), mp.mpf(2))))
        <= tol,
    )
    check(
        # Q(1, y) = e^-y exactly. Checked at y = 700, far right of the mean
        # (so the continued-fraction branch runs) and small enough that the
        # lead guard does not fire -- the branch that would otherwise only be
        # exercised by real sweep rows.
        "gamma_upper.exponential_closed_form_far_right",
        abs(_gamma_upper(mp.mpf(1), mp.mpf(700)) - mp.e ** mp.mpf(-700))
        <= mp.mpf("1e-45") * mp.e ** mp.mpf(-700),
    )

    # --- _phi_diff / _norm_q (truncated_normal's survival-form pieces) ---
    check(
        "phi_diff.full_line_is_one",
        abs(_phi_diff(mp.mpf("-inf"), mp.mpf("inf")) - 1) <= tol,
    )
    check(
        # The #57 seam: a same-tail window where a Phi difference cancels.
        # Anchored on the tail identity Z = Q(5) - Q(6), which _phi_diff must
        # reproduce to full precision rather than to the ~7 digits a
        # Phi-difference would leave.
        "phi_diff.same_tail_window_survival_identity",
        abs(_phi_diff(mp.mpf(5), mp.mpf(6)) - (_norm_q(mp.mpf(5)) - _norm_q(mp.mpf(6))))
        <= mp.mpf("1e-45") * _phi_diff(mp.mpf(5), mp.mpf(6)),
    )
    check(
        # Reflection: the left-tail branch and the right-tail branch must
        # agree on mirrored windows (they are different code paths).
        "phi_diff.reflection_symmetry",
        abs(_phi_diff(mp.mpf(-6), mp.mpf(-5)) - _phi_diff(mp.mpf(5), mp.mpf(6)))
        <= mp.mpf("1e-45") * _phi_diff(mp.mpf(5), mp.mpf(6)),
    )

    # --- _std_normal_quantile (survival-form, no 2p-1 cancellation) ---
    check(
        # Tolerance is 1e-40, not the 1e-45 the solver converges x to: a
        # round trip through Q AMPLIFIES by t^2, since dQ/dt / Q = -t and the
        # solver's own error is relative in t. At t ~ 37 that is 37^2 * 1e-45
        # ~ 1.4e-42, and the observed figure is 5.8e-43. Setting 1e-45 here
        # would be asserting a precision the identity cannot carry, not a
        # tighter oracle.
        "std_normal_quantile.inverts_at_1e-300",
        abs(_norm_q(-_std_normal_quantile(mp.mpf("1e-300"))) - mp.mpf("1e-300"))
        <= mp.mpf("1e-40") * mp.mpf("1e-300"),
    )
    check(
        # Two-sided, per the repo's regression-guard rule: assert the erfinv
        # form is the one that FAILS here, so this check cannot pass on both
        # sides of the defect it guards against.
        "std_normal_quantile.beats_erfinv_form_in_deep_tail",
        mp.isfinite(_std_normal_quantile(mp.mpf("1e-300")))
        and not mp.isfinite(SQRT2 * mp.erfinv(2 * mp.mpf("1e-300") - 1)),
    )
    check(
        # #157: the Gaussian reference itself goes through the survival form,
        # finite and at -37.0471 sigma at p = 1e-300 (the erfinv form gave -inf).
        "gaussian_ref.quantile_finite_at_1e-300",
        abs(GaussianRef.quantile(0.0, 1.0, 1e-300) - mp.mpf("-37.047096299361199"))
        < mp.mpf("1e-12"),
    )

    # --- comparator overflow rule (references beyond double range) ---
    check(
        "overflow_rule.inf_is_correctly_rounded_beyond_double_range",
        _overflow_correct(mp.mpf("1e400"), math.inf),
    )
    check(
        "overflow_rule.sign_must_match",
        not _overflow_correct(mp.mpf("1e400"), -math.inf)
        and _overflow_correct(-mp.mpf("1e400"), -math.inf),
    )
    check(
        # Two-sided: a library value that FAILED to overflow is still flagged.
        "overflow_rule.finite_library_value_still_flags",
        not _overflow_correct(mp.mpf("1e400"), 1.7976931348623157e308),
    )
    check(
        # DBL_MAX itself is below the rounds-to-inf midpoint; 2^1024 is above.
        "overflow_rule.threshold_is_binary64_rounding_midpoint",
        not _overflow_correct(mp.mpf("1.7976931348623157e308"), math.inf)
        and _overflow_correct(mp.mpf(2) ** 1024, math.inf),
    )

    # --- logistic ---
    check("logistic.cdf(mu)==0.5", LogisticRef.cdf(3.0, 2.0, 3.0) == mp.mpf("0.5"))
    check(
        "logistic.quantile_inverts_cdf",
        abs(LogisticRef.quantile(0.0, 1.0, float(LogisticRef.cdf(0.0, 1.0, 2.5))) - mp.mpf(2.5))
        <= mp.mpf("1e-15"),
    )

    # --- gumbel ---
    check(
        "gumbel.cdf(mu)==exp(-1)",
        abs(GumbelRef.cdf(1.0, 3.0, 1.0) - mp.e ** -1) <= tol,
    )
    check(
        "gumbel.median_offset_matches_header_constant",
        # gumbel.h pins kMedianOffset = -log(log 2) = 0.36651292058166433.
        abs(GumbelRef.quantile(0.0, 1.0, 0.5) - mp.mpf("0.36651292058166433"))
        <= mp.mpf("1e-16"),
    )

    # --- bernoulli ---
    check("bernoulli.pmf_sums_to_one", BernoulliRef.pdf(0.3, 0, 0) + BernoulliRef.pdf(0.3, 0, 1) == 1)
    check(
        # bernoulli.h delegates to Binomial(n=1, p), so the frozen BinomialRef
        # is an independent anchor for the whole step, not a restatement of
        # this class's own arithmetic. (Comparing cdf(0) to the literal
        # mp.mpf("0.7") would instead assert that the double 0.3 is decimal
        # 0.3, which it is not -- 1 - mpf(0.3) differs at 1.1e-17.)
        "bernoulli.matches_binomial_n1",
        all(
            BernoulliRef.pdf(0.3, 0, k) == BinomialRef.pdf(1, 0.3, k)
            and BernoulliRef.cdf(0.3, 0, k) == BinomialRef.cdf(1, 0.3, k)
            for k in (0, 1)
        )
        and BernoulliRef.cdf(0.3, 0, -1) == 0,
    )

    # --- erlang ---
    check(
        "erlang.k1_reduces_to_exponential",
        abs(ErlangRef.cdf(1.0, 2.0, 0.75) - ExponentialRef.cdf(2.0, 0, 0.75)) <= tol,
    )
    check(
        "erlang.equals_gamma_at_integer_shape",
        abs(ErlangRef.cdf(4.0, 1.5, 2.0) - GammaRef.cdf(4.0, 1.5, 2.0)) <= tol,
    )

    # --- fisher_f ---
    check(
        # F(d1, d2) at x = 1 with d1 == d2 is the exact symmetry point.
        "fisher_f.cdf(1)==0.5_when_d1==d2",
        abs(FisherFRef.cdf(7.0, 7.0, 1.0) - mp.mpf("0.5")) <= tol,
    )
    check(
        # Reciprocal identity: if X ~ F(d1, d2) then 1/X ~ F(d2, d1), i.e.
        # F_{d1,d2}(x) = 1 - F_{d2,d1}(1/x). Independent of how either side
        # is steered, so it also checks the y / yc split does not skew.
        # x = 2 and 1/x = 0.5 are BOTH exact doubles: at x = 2.5 the check
        # fails at 1.3e-17, which is not the oracle -- it is the two sides
        # being evaluated at different points, since the double nearest 0.4
        # is 0.4 + 2.2e-17 and the pdf there is ~0.58.
        "fisher_f.reciprocal_identity",
        abs(FisherFRef.cdf(5.0, 10.0, 2.0) - (1 - FisherFRef.cdf(10.0, 5.0, 0.5))) <= tol,
    )
    check(
        "fisher_f.quantile_inverts_cdf_both_halves",
        abs(FisherFRef.quantile(5.0, 10.0, float(FisherFRef.cdf(5.0, 10.0, 0.4))) - mp.mpf(0.4))
        <= mp.mpf("1e-14")
        and abs(FisherFRef.quantile(5.0, 10.0, float(FisherFRef.cdf(5.0, 10.0, 4.0))) - mp.mpf(4.0))
        <= mp.mpf("1e-13"),
    )

    # --- inverse_gamma ---
    check(
        # X ~ InvGamma(alpha, scale beta) <=> 1/X ~ Gamma(alpha, rate beta),
        # so F_inv(x) = 1 - F_gamma(1/x). This is the parameterization claim
        # of inverse_gamma.h (beta passed through unchanged), asserted rather
        # than trusted.
        # x = 0.5 and 1/x = 2 are both exact doubles, so the two sides are
        # evaluated at the same mathematical point (see fisher_f above).
        "inverse_gamma.reciprocal_of_gamma",
        abs(InverseGammaRef.cdf(3.0, 2.0, 0.5) - (1 - GammaRef.cdf(3.0, 2.0, 2.0))) <= tol,
    )
    check(
        "inverse_gamma.quantile_inverts_cdf_both_halves",
        abs(
            InverseGammaRef.quantile(3.0, 2.0, float(InverseGammaRef.cdf(3.0, 2.0, 0.5)))
            - mp.mpf(0.5)
        )
        <= mp.mpf("1e-14")
        and abs(
            InverseGammaRef.quantile(3.0, 2.0, float(InverseGammaRef.cdf(3.0, 2.0, 3.0)))
            - mp.mpf(3.0)
        )
        <= mp.mpf("1e-13"),
    )

    # --- half_normal ---
    check(
        # HalfNormal(sigma) is |N(0, sigma)|, so F_HN(x) = 2*Phi(x/sigma) - 1.
        "half_normal.is_folded_gaussian",
        abs(HalfNormalRef.cdf(2.0, 0, 1.5) - (2 * GaussianRef.cdf(0.0, 2.0, 1.5) - 1)) <= tol,
    )
    check(
        "half_normal.quantile_inverts_cdf_deep_tail",
        abs(
            HalfNormalRef.cdf(2.0, 0, float(HalfNormalRef.quantile(2.0, 0, 1e-300)))
            / mp.mpf("1e-300")
            - 1
        )
        <= mp.mpf("1e-14"),
    )

    # --- truncated_normal ---
    check(
        # Degenerate window: the doubly-infinite instance must BE the plain
        # Gaussian, checked against the frozen GaussianRef rather than
        # against itself.
        "truncated_normal.infinite_window_is_gaussian",
        abs(TruncatedNormalRef.cdf(1000.0, 100.0, 1150.0) - GaussianRef.cdf(1000.0, 100.0, 1150.0))
        <= tol,
    )
    check(
        # Same-tail window (the #57 seam): the CDF must hit 0 and 1 at the
        # window edges and 0.5 at the median, with Z ~ 2.86e-7 -- a
        # Phi-difference reference loses ~7 digits building this.
        "truncated_normal.same_tail_window_endpoints_and_median",
        TruncatedNormalRef.cdf(0.0, 2.0, 10.0) == 0
        and TruncatedNormalRef.cdf(0.0, 2.0, 12.0) == 1
        and abs(
            TruncatedNormalRef.cdf(
                0.0, 2.0, float(TruncatedNormalRef.quantile(0.0, 2.0, 0.5))
            )
            - mp.mpf("0.5")
        )
        <= mp.mpf("1e-15"),
    )
    check(
        # The density must integrate to 1 over the window -- an independent
        # quadrature check on the normalization, not a restatement of it.
        # The mpf nodes are passed through UNCAST: rounding them to double
        # (float(t)) starves mpmath's tanh-sinh rule of the precision it
        # needs and the integral lands around 1e-17 off, which reads as an
        # oracle failure and is not one.
        "truncated_normal.density_integrates_to_one",
        abs(mp.quad(lambda t: TruncatedNormalRef.pdf(0.0, 1.0, t), [-2, 0, 2]) - 1)
        <= mp.mpf("1e-25"),
    )
    check(
        "truncated_normal.unknown_window_raises",
        _tn_window_raises_on_unknown_key(),
    )

    _hardening_self_checks(check)
    return results


def _next_double(v: "mp.mpf", k: int = 1) -> "mp.mpf":
    """The double k steps above the correctly rounded double of v."""
    d = float(v)
    for _ in range(k):
        d = math.nextafter(d, math.inf)
    return mp.mpf(d)


def _hardening_self_checks(check) -> None:
    """Checks added with the 2026-10-11 hardening: scorer negative controls,
    the repaired references, and large-shape anchors against independent
    quadrature. Each two-sided where the defect it guards has a concrete
    former form: the check asserts the former form FAILS it."""
    half_ulp = TWO_POW_NEG52 / 2
    quarter_ulp = TWO_POW_NEG52 / 4

    # --- scorer: a 1-ULP perturbation is detected at every magnitude ---
    # The correctly rounded double scores at most half an ULP, the next double
    # up at least a quarter (half an ULP at the reference's own spacing, which
    # is 2^-53..2^-52 relative in the normal range). Below DBL_MIN the former
    # absolute rule scored the same perturbation below 1e-290: two-sided.
    for ref in (mp.mpf("0.3"), mp.mpf("1e-300"), mp.mpf(2) ** -1022 * mp.mpf("1.5"),
                mp.mpf("3e-310"), mp.mpf("7.3e-322"), mp.mpf("1e-330")):
        rounded = mp.mpf(float(ref))
        bumped = _next_double(ref)
        check(f"scorer.correctly_rounded_within_half_ulp({mp.nstr(ref, 3)})",
              relerr(rounded, ref) <= half_ulp)
        check(f"scorer.one_ulp_perturbation_detected({mp.nstr(ref, 3)})",
              relerr(bumped, ref) >= quarter_ulp)
        if ref < _DBL_MIN:
            check(f"scorer.former_absolute_rule_was_blind({mp.nstr(ref, 3)})",
                  abs(bumped - ref) < mp.mpf("1e-290"))
    check("scorer.below_half_denorm_is_exact_zero",
          relerr(mp.mpf(0), mp.mpf("1e-330")) == 0
          and relerr(mp.mpf(2) ** -1074, mp.mpf("1e-330")) == TWO_POW_NEG52)
    # Deep in the subnormal range the law's own allowance F |ln F| 2^-52 is far
    # below one subnormal ULP, so the floor of one ULP governs: a 1-ULP error
    # is within it, three are flagged. (Just below DBL_MIN the law still
    # allows ~700 ULPs -- 3e-310 allows 9.6 -- falling continuously to 1.)
    _r = mp.mpf("7.3e-322")
    check("scorer.law_frac_subnormal_one_ulp_allowance",
          law_frac(_next_double(_r, 1), _r) <= mp.mpf("1.5")
          and law_frac(_next_double(_r, 3), _r) > 1)

    # --- Cauchy left tail: the tail form keeps full relative precision ---
    _w = 1 / mp.mpf(1e60)  # the double -1e60 is the swept point
    _anchor = (_w - _w**3 / 3) / mp.pi  # atan(w)/pi by its series, w = 1/|z|
    _new = CauchyRef.cdf(0.0, 1.0, -1e60)
    _old = mp.mpf("0.5") + mp.atan(mp.mpf(-1e60)) / mp.pi
    check("cauchy.left_tail_relative_precision",
          abs(_new - _anchor) <= mp.mpf("1e-45") * _anchor
          and abs(_old - _anchor) > mp.mpf("1e-3") * _anchor)
    check("cauchy.right_tail_complement",
          abs(CauchyRef.cdf(0.0, 1.0, 1e60) - (1 - _anchor)) <= mp.mpf("1e-45"))

    # --- von Mises tail: scaled quadrature vs an independent formulation ---
    # F(t) for t <= 0 is int_{-1}^{cos t} e^{kappa u} / sqrt(1 - u^2) du over
    # 2 pi I0(kappa) (u = cos s), integrated as w = 1 + u from 0 so no node
    # lands on the singular endpoint. The former unscaled quotient misses
    # (200, -2.5) -- F = 1.7e-158, inside double range -- by 4e-4 relative:
    # mp.quad's absolute test stops early on an integrand of size e^-160.
    def _vm_cos_form(k, t):
        with mp.workdps(80):
            kk, W = mp.mpf(k), 1 + mp.cos(mp.mpf(t))
            num = mp.quad(lambda w: mp.exp(kk * (w - W)) / mp.sqrt(w * (2 - w)), [0, W])
            r = num * mp.exp(kk * (W - 1)) / (2 * mp.pi * mp.besseli(0, kk))
        return +r

    for _k, _t in ((200.0, -2.5), (1000.0, -2.5), (1e4, -0.2), (1e8, -0.002)):
        _ind = _vm_cos_form(_k, _t)
        _got = _vm_small_side(_k, mp.mpf(_t) + mp.pi)
        check(f"vonmises.tail_vs_cos_substitution(kappa={_k},t={_t})",
              abs(_got - _ind) <= mp.mpf("1e-40") * _ind)
    _k, _t = 200.0, mp.mpf(-2.5)
    _f = lambda q: mp.e ** (mp.mpf(_k) * mp.cos(q))
    _old_vm = mp.quad(_f, [-mp.pi, _t]) / mp.quad(_f, [-mp.pi, 0, mp.pi])
    _ind = _vm_cos_form(_k, _t)
    check("vonmises.former_unscaled_quadrature_fails_the_anchor",
          abs(_old_vm - _ind) > mp.mpf("1e-6") * _ind
          and abs(vonmises_cdf(0.0, _k, -2.5) - _ind) <= mp.mpf("1e-40") * _ind)
    _q = vonmises_quantile(0.0, 1000.0, 1e-100)
    check("vonmises.quantile_inverts_cdf_deep_tail_large_kappa",
          abs(_vm_cdf_t(1000.0, _q) - mp.mpf(1e-100)) <= mp.mpf("1e-30") * mp.mpf(1e-100))

    # --- gamma: large-shape anchors against independent quadrature ---
    # 1e4 runs mp.gammainc (both tails and the mean); 5e4 runs the quadrature
    # path, anchored against mp.gammainc where its series converges.
    for _a, _y in ((1e4, 9.5e3), (1e4, 1e4), (1e4, 1.06e4), (2e4, 1.9e4)):
        _P, _Q = _gamma_pq_quad(_a, _y)
        _small = min(_P, _Q)
        _got = _gamma_cdf(mp.mpf(_a), mp.mpf(1), mp.mpf(_y)) if _P <= _Q else \
            _gamma_upper(mp.mpf(_a), mp.mpf(_y))
        check(f"gamma.gammainc_vs_quadrature(a={_a},y={_y})",
              abs(_got - _small) <= mp.mpf("1e-30") * _small)
    _a, _y = mp.mpf(5e4), mp.mpf(4.8e4)
    check("gamma.quadrature_path_vs_gammainc(a=5e4,y=4.8e4)",
          abs(_gamma_cdf(_a, mp.mpf(1), _y) - mp.gammainc(_a, 0, _y, regularized=True))
          <= mp.mpf("1e-30") * _gamma_cdf(_a, mp.mpf(1), _y))
    _P = _gamma_cdf(_a, mp.mpf(1), _y)
    check("gamma.quadrature_upper_is_complement(a=5e4)",
          abs(_gamma_upper(_a, _y) - (1 - _P)) <= mp.mpf("1e-40"))
    # The far-tail guards return exp(lead) only where lead plus a bound on
    # the omitted factor rounds to 0. The bounds are what makes that sound:
    # lead <= log(value) <= lead + bound, checked here against independent
    # values (gamma P and Q by quadrature, I_x by mp.betainc) on both sides of
    # b = 1, where the betainc bound changes form.
    for _a, _y in ((mp.mpf(300), mp.mpf(150)), (mp.mpf(1e4), mp.mpf(8.4e3))):
        _lead, _bound = _gamma_lower_lead(_a, _y)
        _lp = mp.log(_gamma_pq_quad(_a, _y)[0])
        check(f"gamma.lower_lead_bound_brackets_P(a={mp.nstr(_a, 3)})", _lead <= _lp <= _bound)
    for _a, _y in ((mp.mpf(300), mp.mpf(450)), (mp.mpf(1e4), mp.mpf(1.2e4))):
        _lead, _bound = _gamma_upper_lead(_a, _y)
        _lq = mp.log(_gamma_pq_quad(_a, _y)[1])
        check(f"gamma.upper_lead_bound_brackets_Q(a={mp.nstr(_a, 3)})", _lead <= _lq <= _bound)
    for _a, _b, _x in ((50.0, 0.3, 0.2), (50.0, 3.0, 0.2), (400.0, 0.5, 0.5)):
        _a, _b, _x = mp.mpf(_a), mp.mpf(_b), mp.mpf(_x)
        _lead, _bound = _betainc_lead(_a, _b, _x)
        _li = mp.log(mp.betainc(_a, _b, 0, _x, regularized=True))
        # The lower side: the factor is at least (1 - x)^(b-1) for b > 1, at
        # least 1 for b < 1.
        _floor = _lead + ((_b - 1) * mp.log1p(-_x) if _b > 1 else 0)
        check(f"betainc.lead_bound_brackets_I(b={mp.nstr(_b, 3)})", _floor <= _li <= _bound)

    # --- fallback accounting: a gammainc failure is counted, not hidden ---
    _orig = mp.gammainc
    _before = FALLBACKS["gamma_cdf: gammainc raised, quadrature used"]

    def _raising(*a, **k):
        raise _MPNoConvergence("self-check")

    mp.gammainc = _raising
    try:
        _v = _gamma_cdf(mp.mpf(100), mp.mpf(1), mp.mpf(90))
    finally:
        mp.gammainc = _orig
    _counted = FALLBACKS["gamma_cdf: gammainc raised, quadrature used"] - _before
    FALLBACKS["gamma_cdf: gammainc raised, quadrature used"] = _before
    if not FALLBACKS["gamma_cdf: gammainc raised, quadrature used"]:
        del FALLBACKS["gamma_cdf: gammainc raised, quadrature used"]
    check("fallback.gammainc_failure_counted_and_quadrature_used",
          _counted == 1
          and abs(_v - mp.gammainc(100, 0, 90, regularized=True)) <= mp.mpf("1e-40"))

    # --- non-convergence raises ---
    try:
        _discrete_search_quantile(lambda k: mp.mpf(0), 0, None, mp.mpf("0.5"), 0)
        _raised = False
    except OracleNonConvergence:
        _raised = True
    check("discrete_search.unreachable_p_raises", _raised)
    try:
        _tail_logspace_bisect(lambda xx: xx, mp.mpf("0.5"), mp.mpf(1), max_steps=3)
        _raised = False
    except OracleNonConvergence:
        _raised = True
    check("tail_logspace_bisect.unconverged_raises", _raised)

    # --- Student-t: nu = 1e6 betainc path vs quadrature; nu = 1e18 quadrature
    #     path vs the 1/nu Edgeworth term of the Gaussian limit ---
    for _t in (-3.0, -30.0, 5.0):
        _b = _student_t_cdf(mp.mpf(1e6), mp.mpf(_t))
        _qd = _student_t_cdf_quad(1e6, _t)
        _ref = min(_qd, 1 - _qd)
        check(f"student_t.betainc_vs_quadrature(nu=1e6,t={_t})",
              abs(_b - _qd) <= mp.mpf("1e-30") * _ref)
    _nu, _t = mp.mpf(1e18), mp.mpf(-5)
    _edge = mp.erfc(-_t / SQRT2) / 2 - mp.npdf(_t) * (_t**3 + _t) / (4 * _nu)
    check("student_t.quadrature_vs_gaussian_limit(nu=1e18,t=-5)",
          abs(_student_t_cdf(_nu, _t) - _edge) <= mp.mpf("1e-30") * _edge)

    # --- Fisher F(5000, 5000): incomplete beta (a = b = 2500, continued
    #     fraction) vs quadrature of the density, both tails ---
    for _x in (0.9, 1.1):
        _g = lambda t: FisherFRef.logpdf(5000.0, 5000.0, t)
        with mp.workdps(80):
            if _x < 1:
                _qd = _quad_log_tail(_g, mp.mpf(_x), True, 0, mp.inf, mp.mpf("0.02"))
            else:
                _qd = _quad_log_tail(_g, mp.mpf(_x), False, 0, mp.inf, mp.mpf("0.02"))
        _cdf = FisherFRef.cdf(5000.0, 5000.0, _x)
        _got = _cdf if _x < 1 else 1 - _cdf
        check(f"fisher_f.cdf_vs_quadrature(5000,5000,x={_x})",
              abs(_got - _qd) <= mp.mpf("1e-28") * _qd)

    # --- Fisher F quantile with the root's y near 1 ---
    # F(2, d2) has F(x) = 1 - (d2 / (d2 + 2x))^(d2/2), so the median of
    # F(2, 0.002) is 0.001 (2^1000 - 1) = 1.07e298, with 1 - y = 9.3e-302.
    # Two-sided: the former branch (solve in y because p <= 1/2) gives 2.1e42.
    _d2 = mp.mpf(0.002)
    _x = (_d2 / 2) * (mp.mpf(2) ** (2 / _d2) - 1)  # d2 is the double 0.002
    _y = _tail_logspace_bisect(lambda yy: _betainc_reg(mp.mpf(1), _d2 / 2, yy), mp.mpf("0.5"),
                               1 / (1 + _d2 / 2), hi_cap=1)
    _former = (_d2 / 2) * _y / (1 - _y)
    check("fisher_f.quantile_root_with_y_near_one",
          abs(FisherFRef.quantile(2.0, 0.002, 0.5) - _x) <= mp.mpf("1e-40") * _x
          and abs(_former - _x) > mp.mpf("1e-3") * _x)

    # --- truncated normal, half-infinite windows ---
    # (-inf, -5] at mean 1, sd 1: F(x) = Phi(x - 1) / Phi(-5) from erfc.
    _x = -40.0
    _ind = mp.erfc(-(mp.mpf(_x) - 1) / SQRT2) / mp.erfc(5 / SQRT2)
    check("truncated_normal.left_half_infinite_cdf",
          abs(TruncatedNormalRef.cdf(1.0, 1.0, _x) - _ind) <= mp.mpf("1e-40") * _ind)
    for _key, _p in (((2.0, 1.0), 1e-300), ((2.0, 1.0), 1 - 1e-15), ((0.0, 0.5), 1e-300),
                     ((0.0, 0.5), 1 - 1e-15), ((3.0, 1.0), 0.25), ((4.0, 1.0), 0.999)):
        # The root must lie within 1e-40 relative of the returned quantile: the
        # small-side mass at q (1 -+ 1e-40) brackets p Z. (Inverting to a
        # relative p is ill-conditioned at a window edge: at [30, inf) the
        # p = 1e-300 quantile sits 1.7e-302 above the bound.)
        _q = TruncatedNormalRef.quantile(_key[0], _key[1], _p)
        _a, _bb = _tn_window(*_key)
        _al = (mp.mpf(_a) - _key[0]) / _key[1]
        _be = (mp.mpf(_bb) - _key[0]) / _key[1]
        _Z = _phi_diff(_al, _be)
        _xs = [((_q * (1 + _e * mp.mpf("1e-40") * mp.sign(_q))) - _key[0]) / _key[1]
               for _e in (-1, 1)]
        if _p <= 0.5:
            _m = [_phi_diff(_al, _xi) for _xi in _xs]
            _ok = _m[0] <= mp.mpf(_p) * _Z <= _m[1]
        else:
            _m = [_phi_diff(_xi, _be) for _xi in _xs]
            _ok = _m[1] <= (1 - mp.mpf(_p)) * _Z <= _m[0]
        check(f"truncated_normal.half_infinite_quantile_brackets_root({_key},p={_p})", _ok)


def _tn_window_raises_on_unknown_key() -> bool:
    """The _TN_WINDOWS guard must actually fire: a sweep instance added on the
    C++ side without a window here has to stop the run, not silently
    characterize the wrong distribution."""
    try:
        _tn_window(123.0, 456.0)
    except RuntimeError:
        return True
    return False


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
                    m = re.search(r"isa=(\S+)", meta["banner"])
                    if m:
                        meta["isa"] = m.group(1)
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


_DBL_MIN = mp.mpf(2) ** -1022
_DENORM_MIN = mp.mpf(2) ** -1074


def subnormal_ulps(got: "mp.mpf", ref: "mp.mpf") -> "mp.mpf":
    """|got - ref| in units of the smallest subnormal, 2^-1074. A reference at
    or below half of it rounds to 0 in double (ties to even), so it is scored
    as an exact 0: any nonzero got costs got / 2^-1074 ULPs."""
    if abs(ref) <= _HALF_DENORM_MIN:
        ref = mp.mpf(0)
    return abs(got - ref) / _DENORM_MIN


def relerr(got: "mp.mpf", ref: "mp.mpf") -> "mp.mpf":
    """Relative error down to DBL_MIN. Below it the reference is subnormal (or
    rounds to 0) and relative error stops measuring rounding, so the error is
    the ULP distance in subnormal units, mapped onto the relative scale as
    ulps * 2^-52: one subnormal ULP scores like one ULP at DBL_MIN, and the
    scale is continuous across DBL_MIN.

    Until 2026-10-11 every reference below 1e-290 was scored by ABSOLUTE
    error, so a result wrong in every digit there (a 1e-300 for a 3e-305, a
    subnormal off by 10^4 ULPs) scored ~1e-300 and never showed."""
    if abs(ref) >= _DBL_MIN:
        return abs(got - ref) / abs(ref)
    return subnormal_ulps(got, ref) * TWO_POW_NEG52


def law_frac(got: "mp.mpf", ref: "mp.mpf") -> "mp.mpf":
    """Observed error as a fraction of the deep-tail CDF law (law_of_f). For a
    subnormal or zero reference the law's absolute error F |ln F| 2^-52 is far
    below one subnormal ULP, so the allowance is max(1 ULP, that), in ULPs:
    a faithfully rounded result scores at most 1."""
    if abs(ref) >= _DBL_MIN:
        return relerr(got, ref) / law_of_f(ref)
    allowed = max(mp.mpf(1), law_of_f(ref) * abs(ref) / _DENORM_MIN)
    return subnormal_ulps(got, ref) / allowed


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
            lf = law_frac(got, ref)
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


# Overflow threshold of IEEE binary64 under round-to-nearest-even: any real
# with |v| >= 2^1024 - 2^970 (the midpoint above DBL_MAX; ties-to-even sends
# the midpoint itself up) rounds to +-inf, so a library +-inf of the
# reference's sign IS the correctly rounded double there — score it correct,
# not as a "reference is finite" contract violation. The v2.4.0 extension's
# fisher_f pdf/quantile and inverse_gamma quantile references run 2.5e+317
# ... 7.4e+2939; the mpf-finiteness test used before 2026-09-03 flagged all
# eight such rows as violations. References in (DBL_MAX, midpoint) still
# take the relative-error path: the correctly rounded double there is
# DBL_MAX, and a library inf would be a genuine one-step overflow error.
with mp.workprec(80):
    _DBL_ROUNDS_TO_INF = mp.mpf(2) ** 1024 - mp.mpf(2) ** 970
    # Half the smallest subnormal: references at or below it round to 0.
    _HALF_DENORM_MIN = mp.mpf(2) ** -1075


def _overflow_correct(ref, got: float) -> bool:
    """True iff |ref| rounds to double +-inf and got is that infinity."""
    return abs(ref) >= _DBL_ROUNDS_TO_INF and math.isinf(got) and (got > 0) == (ref > 0)


def compare(rows: list) -> "tuple[dict, list]":
    groups: dict = {}
    skipped = []

    n_done = 0
    for row in rows:
        n_done += 1
        if n_done % 500 == 0:
            print(f"  [progress] {n_done}/{len(rows)} rows", file=sys.stderr, flush=True)
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

        if inf_input and row.method in ("pdf", "logpdf", "cdf"):
            # +-inf inputs have universal limits for every distribution on
            # (a subset of) the real line: pdf -> 0, logpdf -> -inf,
            # cdf -> 1 at +inf / 0 at -inf. Evaluating the raw reference
            # formulas AT the limit instead produces inf-inf / 0*inf NaN
            # artifacts, which is an oracle artifact, not a finding. Von
            # Mises is periodic -- no limit exists, the library's NaN is
            # the right answer, so check NaN agreement like NaN-input rows.
            if row.dist == "von_mises":
                # Periodic distribution: no mathematical limit exists at
                # +-inf, so no reference value can be asserted. The library
                # uses a deliberate saturation convention (pdf -> 0,
                # logpdf -> -inf, cdf -> 0/1), which the characterization
                # doc records as a convention; the only checkable contract
                # here is that scalar and batch agree with each other.
                if row.batch is not None and not (
                    (is_nan(row.scalar) and is_nan(row.batch))
                    or row.scalar == row.batch
                ):
                    g.violations.append((row.lineno, "batch",
                        f"scalar/batch disagree at +-inf input "
                        f"({row.scalar} vs {row.batch})"))
                continue
            if row.method == "pdf":
                ref = mp.mpf(0)
            elif row.method == "logpdf":
                ref = mp.mpf("-inf")
            else:
                ref = mp.mpf(1) if row.x > 0 else mp.mpf(0)
        else:
            try:
                _t0 = time.perf_counter()
                ref = fn(row.p1, row.p2, row.x)
                _dt = time.perf_counter() - _t0
                if _dt > 1.0:
                    print(f"  [slow-row] line {row.lineno} {row.dist}/{row.method} "
                          f"p1={row.p1!r} p2={row.p2!r} x={row.x!r} took {_dt:.1f}s",
                          file=sys.stderr, flush=True)
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
            if mp.isnan(ref):
                # A NaN reference on a non-inf input row is an oracle bug,
                # not a library finding; surface it as such.
                g.violations.append(
                    (row.lineno, "oracle", f"oracle produced NaN reference (x={row.x!r})")
                )
                continue
            scalar_ok = math.isinf(row.scalar) and (row.scalar > 0) == (ref > 0)
            if not scalar_ok:
                g.violations.append(
                    (row.lineno, "scalar", f"reference is {ref}, scalar_bits decoded to {row.scalar}")
                )
            if row.batch is not None:
                batch_ok = math.isinf(row.batch) and (row.batch > 0) == (ref > 0)
                if not batch_ok:
                    g.violations.append(
                        (row.lineno, "batch", f"reference is {ref}, batch_bits decoded to {row.batch}")
                    )
            continue

        if abs(ref) >= _DBL_ROUNDS_TO_INF:
            # Finite in mpf but beyond double range: the correctly rounded
            # double is +-inf. Two-sided — a finite library value here is
            # still flagged (it failed to overflow).
            if not _overflow_correct(ref, row.scalar):
                g.violations.append(
                    (row.lineno, "scalar",
                     f"reference {mp.nstr(ref, 6)} rounds to double +-inf, "
                     f"scalar_bits decoded to {row.scalar}")
                )
            if row.batch is not None and not _overflow_correct(ref, row.batch):
                g.violations.append(
                    (row.lineno, "batch",
                     f"reference {mp.nstr(ref, 6)} rounds to double +-inf, "
                     f"batch_bits decoded to {row.batch}")
                )
            continue

        if row.method == "quantile" and abs(ref) > _HALF_DENORM_MIN:
            # An exact 0 against a non-zero reference is a collapse, not a
            # small absolute error: LogNormal's quantile returned 0 at
            # p = 1e-300 against references down to 6.6e-33, and the absolute
            # gate passed it (#157, #158). References below half the smallest
            # subnormal round to 0, so a 0 there is correct.
            for src_name, v in (("scalar", row.scalar), ("batch", row.batch)):
                if v is not None and v == 0:
                    g.violations.append(
                        (row.lineno, src_name,
                         f"reference is {mp.nstr(ref, 6)}, {src_name} returned exactly 0")
                    )

        is_cdf_tail = row.method == "cdf" and ref < mp.mpf("1e-3")
        if not math.isfinite(row.scalar):
            g.violations.append(
                (row.lineno, "scalar", f"reference is finite ({mp.nstr(ref, 6)}), scalar_bits decoded to {row.scalar}")
            )
            continue
        g.scalar.observe(mp.mpf(row.scalar), ref, row.x, is_cdf_tail)

        if row.batch is not None and not math.isfinite(row.batch):
            g.violations.append(
                (row.lineno, "batch", f"reference is finite ({mp.nstr(ref, 6)}), batch_bits decoded to {row.batch}")
            )
        elif row.batch is not None:
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


def print_fallbacks() -> None:
    """Every reference built by a helper's secondary method, by kind."""
    total = sum(FALLBACKS.values())
    print(f"oracle fallbacks: {total}")
    for kind, n in sorted(FALLBACKS.items()):
        print(f"  {n:>6}  {kind}")


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
                    f"| {method} | *(contract)* | {len(g.violations)} violation(s) -- see appendix | | | | | |"
                )
        lines.append("")
    # Durable appendix: every contract violation, so the checked-in doc
    # stands alone without the stdout log.
    all_v = [
        (dist, method, lineno, who, msg)
        for (dist, method), g in sorted(groups.items())
        for (lineno, who, msg) in g.violations
    ]
    if all_v:
        lines.append("### Contract findings (appendix)" + "\n")
        lines.append(
            f"{len(all_v)} contract violations across the sweep. `csv_line` "
            "indexes the sweep CSV this report was generated from (see the "
            "commit/isa banner in the regeneration log)."
        )
        lines.append("")
        lines.append("| dist | method | source | csv_line | finding |")
        lines.append("|---|---|---|---|---|")
        for dist, method, lineno, who, msg in all_v:
            lines.append(f"| {dist} | {method} | {who} | {lineno} | {msg} |")
        lines.append("")
    return "\n".join(lines)


# One generated block per ISA, so a sweep from a second machine adds a
# block instead of overwriting the first. The ISA label is the sweep
# banner's `isa=` value (SIMDPolicy::getLevelString(): "AVX-512", "AVX2",
# "AVX", "SSE2", "NEON"). The unlabelled `<!-- BEGIN GENERATED -->` form is
# the pre-2026-08-23 single-machine layout; it is refused rather than
# silently overwritten -- relabel it by hand once with the ISA it holds.
GEN_BEGIN_FMT = "<!-- BEGIN GENERATED isa={isa} -->"
GEN_END_FMT = "<!-- END GENERATED isa={isa} -->"
GEN_END_ANY = re.compile(r"^<!-- END GENERATED isa=\S+ -->$", re.M)
GEN_LEGACY = "<!-- BEGIN GENERATED -->"


def _marker_re(marker: str) -> "re.Pattern[str]":
    # Markers count only as whole lines, so prose that quotes a marker (the
    # doc's own "Regenerating" section does) is not mistaken for one.
    return re.compile("^" + re.escape(marker) + "$", re.M)


def rewrite_doc(doc_path: str, generated_body: str, isa: str) -> str:
    """Replace the `isa` block in doc_path, or append a new one after the last
    existing labelled block. Returns "replaced" or "added"."""
    with open(doc_path, "r", encoding="utf-8") as f:
        text = f.read()
    if _marker_re(GEN_LEGACY).search(text):
        raise ValueError(
            f"{doc_path}: found the unlabelled {GEN_LEGACY} block; relabel it "
            f"as {GEN_BEGIN_FMT.format(isa='<ISA>')} / {GEN_END_FMT.format(isa='<ISA>')} "
            "with the ISA it was generated on before regenerating"
        )
    begin = GEN_BEGIN_FMT.format(isa=isa)
    end = GEN_END_FMT.format(isa=isa)
    block = begin + "\n\n" + generated_body + "\n" + end
    mb, me = _marker_re(begin).search(text), _marker_re(end).search(text)
    if mb:
        if not me or me.start() < mb.end():
            raise ValueError(f"{doc_path}: {begin} without matching {end}")
        new_text = text[: mb.start()] + block + text[me.end() :]
        outcome = "replaced"
    else:
        ends = list(GEN_END_ANY.finditer(text))
        if not ends:
            raise ValueError(f"{doc_path}: no labelled GENERATED block to append after")
        at = ends[-1].end()
        new_text = text[:at] + "\n\n" + block + text[at:]
        outcome = "added"
    with open(doc_path, "w", encoding="utf-8") as f:
        f.write(new_text)
    return outcome


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
    parser.add_argument(
        "--isa",
        default=None,
        help="ISA label for the generated block (default: the sweep banner's isa= value)",
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
    print_fallbacks()

    isa = args.isa or meta.get("isa")
    if not isa:
        print("No isa= in the sweep banner and no --isa given; cannot label the generated block", file=sys.stderr)
        return 1
    # A filtered CSV would replace the whole ISA block with the few distributions it carries, so
    # the default doc is written only from a sweep that covers every distribution.
    missing = sorted(set(DIST_PARAMS) - {row.dist for row in rows})
    if missing and not args.out:
        print(f"\nPartial sweep (missing {', '.join(missing)}): {doc_path} not updated. "
              "Pass --out to write the partial block elsewhere.")
        return 0
    banner = meta.get("banner", "(no banner)")
    body = f"## Generated tables: {isa}\n\nSweep banner: `{banner}`\n\n" + render_markdown(groups)
    outcome = rewrite_doc(doc_path, body, isa)
    print(f"\n{outcome.capitalize()} generated block isa={isa} in {doc_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

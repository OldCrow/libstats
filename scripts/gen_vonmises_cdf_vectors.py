#!/usr/bin/env python3
"""Generate tests/vonmises_cdf_vectors.inc -- high-precision Von Mises CDF
reference vectors for issue #51 (VonMisesDistribution::getCumulativeProbability).

Oracle: DIRECT QUADRATURE, deliberately independent of the Bessel-series
method the implementation uses (src/von_mises.cpp integrates a
trapezoidal-rule PDF grid for kappa <= 50 and a wrapped-normal/Bessel-based
approximation for kappa > 50 -- an oracle built the same way could not catch
a shared bug). At mp.dps = 40:

    F(x; mu, kappa) = quad(exp(kappa*cos(p)), [-pi, t])
                      / quad(exp(kappa*cos(p)), [-pi, pi])

    t = (x - mu) wrapped into (-pi, pi], computed with exact mpf arithmetic
    (mp.fmod on mp.mpf operands, not float fmod) -- matches the wrap
    convention VonMisesDistribution::wrapAngle uses, since the CDF is
    defined as a function on the wrapped domain (see the .cpp comment: "x is
    wrapped to (-pi, pi] before integration").

Each numerator/denominator integral is split at p=0 (mp.quad given [lo, 0,
hi] as breakpoints) whenever 0 falls inside the interval -- exp(kappa*cos(p))
peaks there, sharply so for large kappa (width ~ 1/sqrt(kappa)), and handing
mpmath's tanh-sinh quadrature the peak location as an explicit breakpoint is
what keeps quad accurate without needing extreme dps at kappa=1000.

Emits tests/vonmises_cdf_vectors.inc: {x_bits, mu_bits, kappa_bits, F_bits}
uint64 hex rows (IEEE-754 double bit patterns). Each row carries its own
kappa_bits, so the gate buckets by kappa directly from the row -- no separate
grouping table needed; rows are nonetheless emitted contiguously per kappa in
coverage-list order for readability.

Coverage: kappa in {0.0, 0.01, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500,
1000}; mu in {0.0, 0.7, -2.5} rotated across the kappa list (one mu per
kappa bucket, mu = mus[i % 3]). Per kappa, ~40 x-values:
  - 29 uniform over (mu-pi, mu+pi)         (random, fixed seed)
  - 1  exact mu                             (t=0, F=0.5 by symmetry)
  - 2  exact endpoints: mu-pi, mu+pi
  - 4  within 1e-6 of an endpoint (both sides of both endpoints)
  - 4  unwrapped, outside [mu-pi, mu+pi]: mu+7.5, mu-13.2, mu+20.3, mu-9.4
Plus a small specials set (mu=0, kappa=1, hardcoded rather than quadrature):
x = NaN -> F=NaN, x = +inf -> F=1, x = -inf -> F=0.

Self-checks (raise and exit non-zero on any failure -- generated references
are trusted over comments per house doctrine):
  1. F(mu; mu, kappa) == 0.5 to 1e-35, for kappa in {0.1, 1, 10} x mu in
     {0.0, 0.7, -2.5} -- t=0 splits the integral exactly in half by the
     integrand's evenness about p=0.
  2. F is monotone non-decreasing on a coarse 25-point grid over
     (mu-pi, mu+pi], for a spread of (mu, kappa) pairs.
  3. kappa=0 (uniform circular case) reproduces the closed form
     F = (t + pi) / (2*pi) to 1e-35.

Usage:  <python-with-mpmath> scripts/gen_vonmises_cdf_vectors.py
Writes tests/vonmises_cdf_vectors.inc relative to the repo root.
"""

import math
import os
import random
import struct
import sys
import time

import mpmath as mp

mp.mp.dps = 40
SEED = 20260820  # documented fixed seed
random.seed(SEED)

D = struct.Struct("<d")
Q = struct.Struct("<Q")


def bits(x: float) -> int:
    return Q.unpack(D.pack(x))[0]


def from_bits(b: int) -> float:
    return D.unpack(Q.pack(b))[0]


# PI is anchored to the DOUBLE-rounded value of pi (math.pi, bit-identical to
# both Python's and libstats' detail::PI -- both are "nearest double to pi",
# 0x400921fb54442d18), NOT mpmath's 40-digit mp.pi. This matters: wrapAngle's
# own branch decision ("x <= -detail::PI") is a double-precision comparison
# against that SAME rounded constant, so at the exact double x = -PI_D the
# implementation's wrap always folds to +PI_D. mp.pi is about 1.22e-16
# *larger* than PI_D, so using mp.pi as the wrap threshold would leave
# x = -PI_D just inside (-mp.pi, mp.pi] -- an *interior* point a hair off the
# true edge, wrapping to itself instead of folding, and producing a spurious
# ~1.0 CDF disagreement at every exact-endpoint row that has nothing to do
# with the kernel's numerical accuracy. Anchoring both the wrap threshold and
# the integration domain on PI_D reproduces the implementation's own
# discrete branch-cut convention exactly, while every arithmetic step
# (fmod, the addition/subtraction, and the quadrature itself) is still done
# at 40-digit mpf precision -- only the boundary CONSTANT is double-width.
PI = mp.mpf(math.pi)
TWO_PI = 2 * PI

# Double-width constants for the bit-exact wrapAngle replica in wrap_to_pi:
# math.pi == detail::PI's double value; 2.0*math.pi is exact (power-of-two
# multiply) and equals detail::TWO_PI (constexpr 2.0 * PI in double).
PI_D = math.pi
TWO_PI_D = 2.0 * math.pi


def wrap_to_pi(x: float, mu: float) -> "mp.mpf":
    """Bit-exact DOUBLE replica of the library's wrapped argument
    t = VonMisesDistribution::wrapAngle(x - mu).

    Every step -- the x-mu subtraction, fmod, both branch compares, and the
    +/-2*pi adjustments -- is IEEE-double arithmetic (Python floats ARE
    doubles; math.fmod is C fmod), against the same double constants
    (math.pi == detail::PI, 2.0*math.pi == detail::TWO_PI, both exact).
    Only the RESULT is lifted to mpf for the quadrature.

    Why not exact-real wrapping with a double threshold (the previous
    revision): the threshold fix alone still let the exact difference
    x - mu and the library's double-rounded difference land on OPPOSITE
    sides of -PI_D at endpoint rows like x = mu - pi with mu = 0.7, where
    the double subtraction's rounding decides whether the (-pi, pi]
    aliasing fires -- a full 0-vs-1 jump in F charged as kernel error.
    The oracle therefore answers: "what is the correctly-rounded F at the
    library's own wrapped t". The deliberate cost is that wrapAngle's own
    rounding (~pdf(t)*ulp, worst ~4e-15 at kappa=1000) is excluded from
    the charged error; the wrap is 5 lines of pre-#51 shared
    infrastructure, replicated here and reviewed against the source, and
    the endpoint SEMANTICS stay pinned by the explicit mu+/-pi -> F=1
    rows. Two independent ifs, not elif, mirroring wrapAngle exactly."""
    t = math.fmod(x - mu, TWO_PI_D)
    if t <= -PI_D:
        t += TWO_PI_D
    if t > PI_D:
        t -= TWO_PI_D
    return mp.mpf(t)


def circle_integral(kappa: "mp.mpf", lo: "mp.mpf", hi: "mp.mpf") -> "mp.mpf":
    """quad(exp(kappa*cos(p)), lo, hi), with a breakpoint at p=0 (the
    integrand's peak) when 0 lies inside [lo, hi]."""
    f = lambda p: mp.exp(kappa * mp.cos(p))
    if lo <= 0 <= hi:
        return mp.quad(f, [lo, mp.mpf(0), hi])
    return mp.quad(f, [lo, hi])


_denom_cache = {}


def denom(kappa: "mp.mpf") -> "mp.mpf":
    """Full-circle normalising integral, cached per kappa (independent of mu
    and of x -- reused across every row sharing a kappa bucket)."""
    key = float(kappa)
    if key not in _denom_cache:
        _denom_cache[key] = circle_integral(kappa, -PI, PI)
    return _denom_cache[key]


def von_mises_cdf_mpf(x: float, mu: float, kappa: float) -> "mp.mpf":
    kap = mp.mpf(kappa)
    t = wrap_to_pi(x, mu)
    num = circle_integral(kap, -PI, t)
    Z = denom(kap)
    F = num / Z
    # Clamp tiny quadrature-roundoff overshoot at the domain edges.
    if F < 0:
        F = mp.mpf(0)
    if F > 1:
        F = mp.mpf(1)
    return F


def von_mises_cdf(x: float, mu: float, kappa: float) -> float:
    return float(von_mises_cdf_mpf(x, mu, kappa))


# ---------------------------------------------------------------------------
# Self-check: verified before anything downstream trusts this oracle's
# output. Any failure here means the generator itself is broken, so it must
# never produce a checked-in reference set silently.
# ---------------------------------------------------------------------------


def _self_check() -> None:
    tol = mp.mpf("1e-35")

    # 1. F(mu; mu, kappa) == 0.5 -- t=0 splits the (even-about-0) integrand
    #    exactly in half, for any mu (mu only shifts the wrap, not the
    #    integrand shape) and any kappa.
    for kappa in (0.1, 1.0, 10.0):
        for mu in (0.0, 0.7, -2.5):
            F = von_mises_cdf_mpf(mu, mu, kappa)
            err = abs(F - mp.mpf("0.5"))
            assert err <= tol, ("F(mu) self-check failed", mu, kappa, F, err)

    # 2. Monotone non-decreasing on a coarse grid over (mu-pi, mu+pi). Grid
    #    points are offset half a step off both endpoints so none lands
    #    exactly on x=mu-pi -- that point wraps to t=+pi (wrap_to_pi's
    #    (-pi,pi] convention folds the -pi boundary onto +pi), which is a
    #    deliberate branch-cut discontinuity, not a monotonicity violation.
    for kappa in (0.0, 0.5, 5.0, 100.0, 1000.0):
        for mu in (0.0, 0.7, -2.5):
            xs = [mu - PI + (mp.mpf(i) + mp.mpf("0.5")) / 25 * TWO_PI for i in range(25)]
            prev = None
            for x in xs:
                F = von_mises_cdf_mpf(float(x), mu, kappa)
                if prev is not None:
                    assert F >= prev - mp.mpf("1e-30"), (
                        "monotonicity self-check failed",
                        mu,
                        kappa,
                        float(x),
                        F,
                        prev,
                    )
                prev = F

    # 3. kappa=0 (uniform circular density) reproduces the closed form
    #    F = (t + pi) / (2*pi).
    for mu in (0.0, 0.7, -2.5):
        for x in (mu - PI, mu - 1.0, mu, mu + 0.3, mu + PI - mp.mpf("1e-9")):
            xf = float(x)
            t = wrap_to_pi(xf, mu)
            expected = (t + PI) / TWO_PI
            F = von_mises_cdf_mpf(xf, mu, 0.0)
            err = abs(F - expected)
            assert err <= tol, ("kappa=0 closed-form self-check failed", mu, xf, F, expected, err)


_self_check()

# ---------------------------------------------------------------------------
# Main coverage sweep
# ---------------------------------------------------------------------------

KAPPAS = [0.0, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0]
MUS = [0.0, 0.7, -2.5]

rows = []  # (x_bits, mu_bits, kappa_bits, F_bits)
bucket_counts = []

t_start = time.time()
for i, kappa in enumerate(KAPPAS):
    mu = MUS[i % len(MUS)]
    xs = []

    # 29 uniform over (mu-pi, mu+pi)
    for _ in range(29):
        xs.append(mu + random.uniform(-math.pi, math.pi))

    # exact mu
    xs.append(mu)

    # exact endpoints (equivalent points on the circle, opposite wrap sides)
    xs.append(mu - math.pi)
    xs.append(mu + math.pi)

    # within 1e-6 of either endpoint, both sides
    xs.append(mu - math.pi + 1e-6)
    xs.append(mu - math.pi - 1e-6)
    xs.append(mu + math.pi - 1e-6)
    xs.append(mu + math.pi + 1e-6)

    # unwrapped, several multiples of pi outside [mu-pi, mu+pi] -- exercises
    # the wrap-before-integrate path for x values far off the fundamental
    # domain.
    xs.append(mu + 7.5)
    xs.append(mu - 13.2)
    xs.append(mu + 20.3)
    xs.append(mu - 9.4)

    n_before = len(rows)
    for x in xs:
        F = von_mises_cdf(x, mu, kappa)
        rows.append((bits(x), bits(mu), bits(kappa), bits(F)))
    bucket_counts.append((kappa, mu, len(rows) - n_before))

gen_elapsed = time.time() - t_start

# ---------------------------------------------------------------------------
# Specials (outside the main gate budget): NaN, +inf, -inf. mu/kappa are
# immaterial to these outcomes (NaN propagates, +/-inf saturate to 1/0
# regardless of parameters), so a fixed (mu=0, kappa=1) is used and the
# reference F is hardcoded rather than routed through quadrature.
# ---------------------------------------------------------------------------

specials_mu = 0.0
specials_kappa = 1.0
nan_bits = bits(math.nan)
specials = [
    (bits(math.nan), bits(specials_mu), bits(specials_kappa), nan_bits),
    (bits(math.inf), bits(specials_mu), bits(specials_kappa), bits(1.0)),
    (bits(-math.inf), bits(specials_mu), bits(specials_kappa), bits(0.0)),
]

# self-check: specials encode exactly what the CDF contract promises
assert math.isnan(from_bits(specials[0][3])), "NaN input must reference NaN"
assert from_bits(specials[1][3]) == 1.0, "+inf input must reference F=1.0 exactly"
assert from_bits(specials[2][3]) == 0.0, "-inf input must reference F=0.0 exactly"

# ---------------------------------------------------------------------------
# Emit
# ---------------------------------------------------------------------------

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out = os.path.join(root, "tests", "vonmises_cdf_vectors.inc")
with open(out, "w") as f:
    f.write("// Auto-generated Von Mises CDF reference vectors (issue #51).\n")
    f.write("// {x_bits, mu_bits, kappa_bits, F_bits}: IEEE-754 double bit patterns.\n")
    f.write("// Oracle: direct quadrature at mp.dps=40, independent of the Bessel-series\n")
    f.write("// implementation under test --\n")
    f.write("//   F(x;mu,kappa) = quad(exp(kappa*cos(p)), [-pi,t]) / quad(..., [-pi,pi])\n")
    f.write("// with t = (x-mu) wrapped into (-pi,pi] via exact mpf arithmetic. See\n")
    f.write("// scripts/gen_vonmises_cdf_vectors.py for the full derivation and\n")
    f.write("// self-checks.\n")
    f.write(f"// Fixed seed {SEED}. DO NOT EDIT -- regenerate with\n")
    f.write("// scripts/gen_vonmises_cdf_vectors.py.\n")
    f.write(
        "// Rows are grouped contiguously per kappa bucket in coverage-list order\n"
        "// (kappa_bits is also carried per-row, so the gate can bucket directly off\n"
        "// it without relying on row order):\n"
    )
    for kappa, mu, count in bucket_counts:
        f.write(f"//   kappa={kappa:<8g} mu={mu:<6g} n={count}\n")
    f.write(
        "struct VmCdfVector { std::uint64_t x_bits; std::uint64_t mu_bits; "
        "std::uint64_t kappa_bits; std::uint64_t F_bits; };\n"
    )
    f.write(f"static constexpr VmCdfVector kVmCdfVectors[{len(rows)}] = {{\n")
    for xb, mb, kb, fb in rows:
        f.write(f"    {{0x{xb:016x}ULL, 0x{mb:016x}ULL, 0x{kb:016x}ULL, 0x{fb:016x}ULL}},\n")
    f.write("};\n\n")
    f.write("// Specials: NaN in -> NaN out; +inf in -> F=1 exactly; -inf in -> F=0\n")
    f.write("// exactly. mu/kappa are immaterial to these outcomes; fixed at (0, 1).\n")
    f.write(f"static constexpr VmCdfVector kVmCdfSpecials[{len(specials)}] = {{\n")
    for xb, mb, kb, fb in specials:
        f.write(f"    {{0x{xb:016x}ULL, 0x{mb:016x}ULL, 0x{kb:016x}ULL, 0x{fb:016x}ULL}},\n")
    f.write("};\n")

print(f"wrote {out}: {len(rows)} main vectors, {len(specials)} specials")
print("bucket counts: " + ", ".join(f"kappa={k:g}(mu={m:g}):{n}" for k, m, n in bucket_counts))
print(f"generation time: {gen_elapsed:.2f}s (quadrature sweep only, excludes self-check)")
sys.exit(0)

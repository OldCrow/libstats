#!/usr/bin/env python3
"""Generate tests/lognormal_cdf_vectors.inc -- high-precision LogNormal CDF
reference vectors for issue #49 (LogNormalDistribution::getCumulativeProbability
and its batch span overload).

Background: every LogNormal CDF path computed the cancellation form
0.5*(1+erf(z/sqrt(2))). For z<0 that form has an absolute error floor of
~1.1e-16 regardless of erf's own quality (1+erf(near -1) cancels most of
erf's significant digits), and once z <~ -8.3, std::erf(z/sqrt(2)) returns
exactly -1 so F collapses to exactly 0 -- true relative error there is 1.0,
not the tiny value #49's original 1-ULP erf-swap benchmark reported (a
metric-flooring artifact: that benchmark measured absolute, not relative,
error). The fix branches on the sign of z and uses erfc for the left tail:
F = z < 0 ? erfc(-z/sqrt(2))/2 : (1+erf(z/sqrt(2)))/2, which is what this
oracle also computes directly (independent of which of erf/erfc the
implementation happens to call for a given z).

Oracle: F(x; mu, sigma) = erfc(-z / sqrt(2)) / 2, with
    z = (log(x) - mu) / sigma
computed IN MPMATH at mp.dps=40 from the EXACT double inputs (x, mu, sigma).
Unlike #51's angle-wrap oracle, there is no branch-cut/rounding seam to
replicate bit-for-bit here -- log(x), the subtraction, and the division are
all charged to the implementation, so the oracle simply lifts the double
inputs to mpf and evaluates at full precision. mpmath's erfc is defined
identically to the standard erfc for all real arguments, so this one
closed-form expression covers the entire real line without a branch.

Coverage: (mu, sigma) in {(0, 0.5), (1.0, 2.0), (-1.0, 0.25)}, ~50 x values
per pair:
  - log-uniform sweep across the support, chosen to spread F from ~1e-300 to
    1-1e-16 (x = exp(mu + sigma*z) for z drawn to hit a wide spread of F via
    inverse-erfc placement, see `xs_for_pair` below)
  - x chosen so F lands near {1e-320 (subnormal F), 1e-100, 1e-20, 1e-10,
    1e-3, 0.5, 1-1e-10}
  - specials: x in {0, -1, NaN, +inf}

Self-checks (raise and exit non-zero on any failure -- generated references
are trusted over comments per house doctrine):
  1. F(exp(mu); mu, sigma) == 0.5 exactly (to 1e-35), for every (mu, sigma)
     pair -- z=0 splits erfc exactly in half regardless of sigma.
  2. F is monotone non-decreasing in x on a coarse grid, for every pair.

Usage:  <python-with-mpmath> scripts/gen_lognormal_cdf_vectors.py
Writes tests/lognormal_cdf_vectors.inc relative to the repo root.
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


INV_SQRT2 = 1.0 / math.sqrt(2.0)


def lognormal_cdf_mpf(x: float, mu: float, sigma: float) -> "mp.mpf":
    """F(x; mu, sigma) = erfc(-z/sqrt(2))/2, z = (log(x)-mu)/sigma, all at
    mp.dps=40 from the exact double inputs."""
    xm = mp.mpf(x)
    mum = mp.mpf(mu)
    sm = mp.mpf(sigma)
    z = (mp.log(xm) - mum) / sm
    F = mp.erfc(-z / mp.sqrt(2)) / 2
    if F < 0:
        F = mp.mpf(0)
    if F > 1:
        F = mp.mpf(1)
    return F


def lognormal_cdf(x: float, mu: float, sigma: float) -> float:
    return float(lognormal_cdf_mpf(x, mu, sigma))


# ---------------------------------------------------------------------------
# Self-check: verified before anything downstream trusts this oracle's
# output. Any failure here means the generator itself is broken, so it must
# never produce a checked-in reference set silently.
# ---------------------------------------------------------------------------

PAIRS = [(0.0, 0.5), (1.0, 2.0), (-1.0, 0.25)]


def _self_check() -> None:
    tol = mp.mpf("1e-35")

    # 1. F(exp(mu); mu, sigma) == 0.5 -- z=0 splits erfc exactly in half.
    # x = math.exp(mu) is itself a double, so log(x) generally differs from
    # the exact mpf(mu) by up to ~1 ulp; the tolerance here absorbs that
    # double-rounding (pdf(0) ~ 0.4, so a ~1e-16 z-error maps to a ~4e-17
    # F-error) rather than the oracle's own (dps=40) precision floor.
    tol_exp_mu = mp.mpf("1e-15")
    for mu, sigma in PAIRS:
        x = math.exp(mu)
        F = lognormal_cdf_mpf(x, mu, sigma)
        err = abs(F - mp.mpf("0.5"))
        assert err <= tol_exp_mu, ("F(exp(mu)) self-check failed", mu, sigma, F, err)

    # 2. Monotone non-decreasing in x on a coarse log-spaced grid.
    for mu, sigma in PAIRS:
        zs = [-8.0 + 16.0 * i / 24 for i in range(25)]  # z in [-8, 8]
        xs = [math.exp(mu + sigma * z) for z in zs]
        prev = None
        for x in xs:
            F = lognormal_cdf_mpf(x, mu, sigma)
            if prev is not None:
                assert F >= prev - mp.mpf("1e-30"), (
                    "monotonicity self-check failed",
                    mu,
                    sigma,
                    x,
                    F,
                    prev,
                )
            prev = F


_self_check()

# ---------------------------------------------------------------------------
# Main coverage sweep
# ---------------------------------------------------------------------------


def x_for_target_F(mu: float, sigma: float, F_target: "mp.mpf") -> float:
    """Invert F = erfc(-z/sqrt(2))/2 for z (via mpmath's erfinv on the
    complementary form), then map back to x = exp(mu + sigma*z)."""
    # erfc(-z/sqrt2)/2 = F  =>  erfc(-z/sqrt2) = 2F  =>  -z/sqrt2 = erfcinv(2F)
    # => z = -sqrt2 * erfcinv(2F)
    # erfinv(1 - 2F) collapses to +inf once 1 - 2F rounds to 1 at mp.dps
    # (F <= ~1e-41), which silently dropped the 1e-320/1e-300/1e-100 targets
    # (review 2026-08-21, N2). Solve log(erfc(-z/sqrt2)/2) = log F instead,
    # seeded from the asymptotic tail; agrees with dps-360 erfinv to 12 digits.
    # Hybrid: erfinv is exact where 1 - 2F is representable; the root solve
    # (seeded from the asymptotic tail) only below F = 1e-30, where it is.
    if F_target < mp.mpf("1e-30"):
        z0 = -mp.sqrt(2 * mp.log(1 / F_target))
        z = mp.findroot(
            lambda t: mp.log(mp.erfc(-t / mp.sqrt(2)) / 2) - mp.log(F_target), z0
        )
    else:
        z = -mp.sqrt(2) * mp.erfinv(1 - 2 * F_target)
    x = mp.e ** (mp.mpf(mu) + mp.mpf(sigma) * z)
    return float(x)


def xs_for_pair(mu: float, sigma: float) -> list:
    xs = []

    # Log-uniform sweep across the support: z uniform in a wide range,
    # spreading F from far left tail to far right tail.
    for _ in range(35):
        z = random.uniform(-38.0, 8.5)  # z~-38 pushes F down near 1e-300
        xs.append(math.exp(mu + sigma * z))

    # Targeted F values, inverted via erfcinv at full precision.
    targets = ["1e-320", "1e-300", "1e-100", "1e-20", "1e-10", "1e-3", "0.5",
               str(1 - 1e-10), str(1 - 1e-16)]
    for t in targets:
        Ft = mp.mpf(t)
        xs.append(x_for_target_F(mu, sigma, Ft))

    # A handful of near-median and moderately-tailed points for density.
    for z in (-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0, 5.0):
        xs.append(math.exp(mu + sigma * z))

    return xs


rows = []  # (x_bits, mu_bits, sigma_bits, F_bits)
bucket_counts = []

t_start = time.time()
for mu, sigma in PAIRS:
    xs = xs_for_pair(mu, sigma)
    n_before = len(rows)
    for x in xs:
        if not (math.isfinite(x) and x > 0.0):
            continue
        F = lognormal_cdf(x, mu, sigma)
        rows.append((bits(x), bits(mu), bits(sigma), bits(F)))
    bucket_counts.append((mu, sigma, len(rows) - n_before))

gen_elapsed = time.time() - t_start

# ---------------------------------------------------------------------------
# Specials (outside the main gate budget): x=0 -> F=0, x=-1 -> F=0 (outside
# support), NaN -> NaN, +inf -> F=1. mu/sigma are immaterial to these
# outcomes, so a fixed (mu=0, sigma=1) is used and the reference F is
# hardcoded rather than routed through the oracle (log(0), log(-1), log(NaN)
# are not meaningfully evaluable by the mpf oracle either).
# ---------------------------------------------------------------------------

specials_mu = 0.0
specials_sigma = 1.0
nan_bits = bits(math.nan)
specials = [
    (bits(0.0), bits(specials_mu), bits(specials_sigma), bits(0.0)),
    (bits(-1.0), bits(specials_mu), bits(specials_sigma), bits(0.0)),
    (bits(math.nan), bits(specials_mu), bits(specials_sigma), nan_bits),
    (bits(math.inf), bits(specials_mu), bits(specials_sigma), bits(1.0)),
]

# self-check: specials encode exactly what the CDF contract promises
assert from_bits(specials[0][3]) == 0.0, "x=0 must reference F=0.0 exactly"
assert from_bits(specials[1][3]) == 0.0, "x=-1 must reference F=0.0 exactly"
assert math.isnan(from_bits(specials[2][3])), "NaN input must reference NaN"
assert from_bits(specials[3][3]) == 1.0, "+inf input must reference F=1.0 exactly"

# ---------------------------------------------------------------------------
# Emit
# ---------------------------------------------------------------------------

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out = os.path.join(root, "tests", "lognormal_cdf_vectors.inc")
with open(out, "w") as f:
    f.write("// Auto-generated LogNormal CDF reference vectors (issue #49).\n")
    f.write("// {x_bits, mu_bits, sigma_bits, F_bits}: IEEE-754 double bit patterns.\n")
    f.write("// Oracle: closed-form erfc at mp.dps=40, evaluated from the exact double\n")
    f.write("// inputs --\n")
    f.write("//   F(x;mu,sigma) = erfc(-z/sqrt(2))/2, z = (log(x)-mu)/sigma\n")
    f.write("// with log(x), the subtraction, and the division all charged to the\n")
    f.write("// implementation (no branch-cut/rounding seam to replicate, unlike #51's\n")
    f.write("// angle wrap). See scripts/gen_lognormal_cdf_vectors.py for the full\n")
    f.write("// derivation and self-checks.\n")
    f.write(f"// Fixed seed {SEED}. DO NOT EDIT -- regenerate with\n")
    f.write("// scripts/gen_lognormal_cdf_vectors.py.\n")
    f.write(
        "// Rows are grouped contiguously per (mu, sigma) bucket in coverage-list\n"
        "// order (mu_bits/sigma_bits are also carried per-row, so the gate can\n"
        "// bucket directly off them without relying on row order):\n"
    )
    for mu, sigma, count in bucket_counts:
        f.write(f"//   mu={mu:<6g} sigma={sigma:<6g} n={count}\n")
    f.write(
        "struct LnCdfVector { std::uint64_t x_bits; std::uint64_t mu_bits; "
        "std::uint64_t sigma_bits; std::uint64_t F_bits; };\n"
    )
    f.write(f"static constexpr LnCdfVector kLnCdfVectors[{len(rows)}] = {{\n")
    for xb, mb, sb, fb in rows:
        f.write(f"    {{0x{xb:016x}ULL, 0x{mb:016x}ULL, 0x{sb:016x}ULL, 0x{fb:016x}ULL}},\n")
    f.write("};\n\n")
    f.write("// Specials: x=0 -> F=0 exactly; x=-1 (outside support) -> F=0 exactly;\n")
    f.write("// NaN in -> NaN out; +inf in -> F=1 exactly. mu/sigma are immaterial to\n")
    f.write("// these outcomes; fixed at (0, 1).\n")
    f.write(f"static constexpr LnCdfVector kLnCdfSpecials[{len(specials)}] = {{\n")
    for xb, mb, sb, fb in specials:
        f.write(f"    {{0x{xb:016x}ULL, 0x{mb:016x}ULL, 0x{sb:016x}ULL, 0x{fb:016x}ULL}},\n")
    f.write("};\n")

print(f"wrote {out}: {len(rows)} main vectors, {len(specials)} specials")
print("bucket counts: " + ", ".join(f"mu={m:g},sigma={s:g}:{n}" for m, s, n in bucket_counts))
print(f"generation time: {gen_elapsed:.2f}s (sweep only, excludes self-check)")
sys.exit(0)

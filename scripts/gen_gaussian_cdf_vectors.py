#!/usr/bin/env python3
"""Generate tests/gaussian_cdf_vectors.inc -- high-precision Gaussian CDF
reference vectors for the Gaussian instance of issue #49's bug pattern
(GaussianDistribution::getCumulativeProbability and its batch span overload).

Background: GaussianDistribution never routed through detail::normal_cdf
(fixed for #49 in src/math_utils.cpp), and every one of its own CDF paths
computed the cancellation form 0.5*(1+erf(z/sqrt(2))). For z<0 that form has
an absolute error floor of ~1.1e-16 regardless of erf's own quality
(1+erf(near -1) cancels most of erf's significant digits), and once
z <~ -8.3, std::erf(z/sqrt(2)) returns exactly -1 so F collapses to exactly
0 -- true relative error there is 1.0. The fix branches on the sign of z and
uses erfc for the left tail:
F = z < 0 ? erfc(-z/sqrt(2))/2 : (1+erf(z/sqrt(2)))/2, which is what this
oracle also computes directly (independent of which of erf/erfc the
implementation happens to call for a given z).

Oracle: F(x; mean, sigma) = erfc(-z / sqrt(2)) / 2, with
    z = (x - mean) / sigma
computed IN MPMATH at mp.dps=40 from the EXACT double inputs (x, mean,
sigma). As with #49's lognormal oracle there is no branch-cut/rounding seam
to replicate bit-for-bit -- the subtraction and the division are charged to
the implementation, so the oracle simply lifts the double inputs to mpf and
evaluates at full precision. Unlike lognormal there is no log(x) in the
argument transform, so the implementation's achievable accuracy is at least
as good as lognormal's for the same F.

Coverage: (mean, sigma) in {(0, 1), (1.0, 2.0), (-1.0, 0.25)} -- (0, 1)
deliberately first: GaussianDistribution has a dedicated isStandardNormal_
code path (x*INV_SQRT_2, no subtract/divide) in every CDF site, and this
gate must cover it. ~50 x values per pair:
  - uniform-z sweep across the support, chosen to spread F from ~1e-300 to
    1-1e-16 (x = mean + sigma*z)
  - x chosen so F lands near {1e-320 (subnormal-ish), 1e-100, 1e-20, 1e-10,
    1e-3, 0.5, 1-1e-10, 1-1e-16}
  - specials: x in {-inf, NaN, +inf}

Self-checks (raise and exit non-zero on any failure -- generated references
are trusted over comments per house doctrine):
  1. F(mean; mean, sigma) == 0.5 exactly (to 1e-35), for every pair -- x=mean
     is an exact double, the subtraction gives exactly 0, and z=0 splits
     erfc exactly in half regardless of sigma (no log-of-double slack, so
     no loosened tolerance as in the lognormal generator's check 1).
  2. F is monotone non-decreasing in x on a coarse grid, for every pair.

Usage:  <python-with-mpmath> scripts/gen_gaussian_cdf_vectors.py
Writes tests/gaussian_cdf_vectors.inc relative to the repo root.
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


def gaussian_cdf_mpf(x: float, mean: float, sigma: float) -> "mp.mpf":
    """F(x; mean, sigma) = erfc(-z/sqrt(2))/2, z = (x-mean)/sigma, all at
    mp.dps=40 from the exact double inputs."""
    xm = mp.mpf(x)
    mm = mp.mpf(mean)
    sm = mp.mpf(sigma)
    z = (xm - mm) / sm
    F = mp.erfc(-z / mp.sqrt(2)) / 2
    if F < 0:
        F = mp.mpf(0)
    if F > 1:
        F = mp.mpf(1)
    return F


def gaussian_cdf(x: float, mean: float, sigma: float) -> float:
    return float(gaussian_cdf_mpf(x, mean, sigma))


# ---------------------------------------------------------------------------
# Self-check: verified before anything downstream trusts this oracle's
# output. Any failure here means the generator itself is broken, so it must
# never produce a checked-in reference set silently.
# ---------------------------------------------------------------------------

# (0, 1) first: covers the dedicated isStandardNormal_ code path.
PAIRS = [(0.0, 1.0), (1.0, 2.0), (-1.0, 0.25)]


def _self_check() -> None:
    tol = mp.mpf("1e-35")

    # 1. F(mean) == 0.5 -- exact: x=mean is a double, z is exactly 0.
    for mean, sigma in PAIRS:
        F = gaussian_cdf_mpf(mean, mean, sigma)
        err = abs(F - mp.mpf("0.5"))
        assert err <= tol, ("F(mean) self-check failed", mean, sigma, F, err)

    # 2. Monotone non-decreasing in x on a coarse grid.
    for mean, sigma in PAIRS:
        zs = [-8.0 + 16.0 * i / 24 for i in range(25)]  # z in [-8, 8]
        xs = [mean + sigma * z for z in zs]
        prev = None
        for x in xs:
            F = gaussian_cdf_mpf(x, mean, sigma)
            if prev is not None:
                assert F >= prev - mp.mpf("1e-30"), (
                    "monotonicity self-check failed",
                    mean,
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


def x_for_target_F(mean: float, sigma: float, F_target: "mp.mpf") -> float:
    """Invert F = erfc(-z/sqrt(2))/2 for z, then map back to
    x = mean + sigma*z."""
    # erfc(-z/sqrt2)/2 = F  =>  -z/sqrt2 = erfcinv(2F)  =>
    # z = -sqrt2 * erfinv(1 - 2F)
    z = -mp.sqrt(2) * mp.erfinv(1 - 2 * F_target)
    x = mp.mpf(mean) + mp.mpf(sigma) * z
    return float(x)


def xs_for_pair(mean: float, sigma: float) -> list:
    xs = []

    # Uniform-z sweep across the support, spreading F from far left tail to
    # far right tail (z~-38 pushes F down near 1e-316).
    for _ in range(35):
        z = random.uniform(-38.0, 8.5)
        xs.append(mean + sigma * z)

    # Targeted F values, inverted via erfcinv at full precision.
    targets = ["1e-320", "1e-300", "1e-100", "1e-20", "1e-10", "1e-3", "0.5",
               str(1 - 1e-10), str(1 - 1e-16)]
    for t in targets:
        Ft = mp.mpf(t)
        xs.append(x_for_target_F(mean, sigma, Ft))

    # A handful of near-median and moderately-tailed points for density --
    # -1.0 and -0.5 sit inside the batch path's -1 <= w < 0 plain-erf band.
    for z in (-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0, 5.0):
        xs.append(mean + sigma * z)

    return xs


rows = []  # (x_bits, mean_bits, sigma_bits, F_bits)
bucket_counts = []

t_start = time.time()
for mean, sigma in PAIRS:
    xs = xs_for_pair(mean, sigma)
    n_before = len(rows)
    for x in xs:
        if not math.isfinite(x):
            continue
        F = gaussian_cdf(x, mean, sigma)
        rows.append((bits(x), bits(mean), bits(sigma), bits(F)))
    bucket_counts.append((mean, sigma, len(rows) - n_before))

gen_elapsed = time.time() - t_start

# ---------------------------------------------------------------------------
# Specials (outside the main gate budget): -inf -> F=0 exactly, +inf -> F=1
# exactly, NaN -> NaN. The Gaussian support is the whole real line, so
# unlike lognormal there is no outside-support finite x. mean/sigma are
# immaterial to these outcomes; fixed at (0, 1), and the reference F is
# hardcoded rather than routed through the oracle.
# ---------------------------------------------------------------------------

specials_mean = 0.0
specials_sigma = 1.0
nan_bits = bits(math.nan)
specials = [
    (bits(-math.inf), bits(specials_mean), bits(specials_sigma), bits(0.0)),
    (bits(math.nan), bits(specials_mean), bits(specials_sigma), nan_bits),
    (bits(math.inf), bits(specials_mean), bits(specials_sigma), bits(1.0)),
]

# self-check: specials encode exactly what the CDF contract promises
assert from_bits(specials[0][3]) == 0.0, "x=-inf must reference F=0.0 exactly"
assert math.isnan(from_bits(specials[1][3])), "NaN input must reference NaN"
assert from_bits(specials[2][3]) == 1.0, "+inf input must reference F=1.0 exactly"

# ---------------------------------------------------------------------------
# Emit
# ---------------------------------------------------------------------------

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out = os.path.join(root, "tests", "gaussian_cdf_vectors.inc")
with open(out, "w") as f:
    f.write("// Auto-generated Gaussian CDF reference vectors (#49 bug pattern in\n")
    f.write("// GaussianDistribution).\n")
    f.write("// {x_bits, mean_bits, sigma_bits, F_bits}: IEEE-754 double bit patterns.\n")
    f.write("// Oracle: closed-form erfc at mp.dps=40, evaluated from the exact double\n")
    f.write("// inputs --\n")
    f.write("//   F(x;mean,sigma) = erfc(-z/sqrt(2))/2, z = (x-mean)/sigma\n")
    f.write("// with the subtraction and the division charged to the implementation\n")
    f.write("// (no log in the transform, unlike #49's lognormal vectors). See\n")
    f.write("// scripts/gen_gaussian_cdf_vectors.py for the full derivation and\n")
    f.write("// self-checks.\n")
    f.write(f"// Fixed seed {SEED}. DO NOT EDIT -- regenerate with\n")
    f.write("// scripts/gen_gaussian_cdf_vectors.py.\n")
    f.write(
        "// Rows are grouped contiguously per (mean, sigma) bucket in coverage-list\n"
        "// order (mean_bits/sigma_bits are also carried per-row, so the gate can\n"
        "// bucket directly off them without relying on row order):\n"
    )
    for mean, sigma, count in bucket_counts:
        f.write(f"//   mean={mean:<6g} sigma={sigma:<6g} n={count}\n")
    f.write(
        "struct GaussCdfVector { std::uint64_t x_bits; std::uint64_t mean_bits; "
        "std::uint64_t sigma_bits; std::uint64_t F_bits; };\n"
    )
    f.write(f"static constexpr GaussCdfVector kGaussCdfVectors[{len(rows)}] = {{\n")
    for xb, mb, sb, fb in rows:
        f.write(f"    {{0x{xb:016x}ULL, 0x{mb:016x}ULL, 0x{sb:016x}ULL, 0x{fb:016x}ULL}},\n")
    f.write("};\n\n")
    f.write("// Specials: -inf -> F=0 exactly; NaN in -> NaN out; +inf -> F=1 exactly.\n")
    f.write("// mean/sigma are immaterial to these outcomes; fixed at (0, 1).\n")
    f.write(f"static constexpr GaussCdfVector kGaussCdfSpecials[{len(specials)}] = {{\n")
    for xb, mb, sb, fb in specials:
        f.write(f"    {{0x{xb:016x}ULL, 0x{mb:016x}ULL, 0x{sb:016x}ULL, 0x{fb:016x}ULL}},\n")
    f.write("};\n")

print(f"wrote {out}: {len(rows)} main vectors, {len(specials)} specials")
print("bucket counts: " + ", ".join(f"mean={m:g},sigma={s:g}:{n}" for m, s, n in bucket_counts))
print(f"generation time: {gen_elapsed:.2f}s (sweep only, excludes self-check)")
sys.exit(0)

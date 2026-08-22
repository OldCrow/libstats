#!/usr/bin/env python3
"""Generate tests/trig_ulp_vectors.inc -- correctly-rounded cos()/sin()
reference vectors for the x86 vector_cos_{sse2,avx,avx2,avx512} kernels and
their forthcoming vector_sin_* counterparts (issue #95).

Ported from libhmm's scripts/gen_trig_ulp_vectors.py (same owner, MIT; #95 is
a port-back of libhmm's #74 clean-room quadrant-reduction kernel, already
validated max 1 ULP on Zen 4 -- see PLAN.md). libstats' NEON tier already
carries this derivation (docs/NEON_TRIG_DERIVATION.md, D_max = 2^23); this
generator backs the x86 gate that currently has no fixed kernel to validate
against -- see tests/test_trig_ulp_gates.cpp for the staged
LIBSTATS_TRIG_GATES_HAVE_SIN toggle.

Backs tests/test_trig_ulp_gates.cpp (the per-tier ULP gate for
stats::arch::simd::VectorOps::vector_cos_<tier> / vector_sin_<tier>). Each
entry is (input_bits, cos_bits, sin_bits): cos/sin evaluated at 320-bit
precision with mpmath, each rounded once to nearest double.
Architecture-neutral (pure mathematics) -- the same reference set gates every
ISA tier.

Buckets (main set, all inside the supported vectorized domain |x| <= 2^23,
matching docs/NEON_TRIG_DERIVATION.md's D_max):
  - uniform in [-2pi, 2pi] (the von Mises range)
  - uniform in [-1e4, 1e4]
  - log-uniform magnitude out to +/-2^23 (domain-wide coverage)
  - near k*pi/2 stress walk: for random k up to floor(2^23 * 2/pi), the
    nearest double to k*pi/2 plus a few nextafter neighbours either side.
    Both odd k (cos ~ 0, sin ~ +/-1) and even k (sin ~ 0, cos ~ +/-1) are
    included so reduction error is stressed for both functions' near-zero
    outputs.

Plus a separate small kTrigUlpSpecials array (same struct, outside the main
gate budget): +/-0.0, +/-D_MAX exactly, a few values just above D_MAX
(exercising the batch wrappers' scalar-libm fixup path), and +/-inf, NaN
(reference bits: cos/sin of +/-inf and NaN are NaN, so those four entries are
hardcoded rather than routed through mpmath).

Self-check: verifies a handful of known closed-form values (cos(0)=1,
sin(0)=0, cos(pi)=-1, sin(pi/2)=1, cos(pi/2)~0) before trusting mpmath's
output, and asserts every main-bucket point respects the domain bound and
every NaN/Inf special is encoded as IEEE-754 predicts. Per house doctrine
(generated reference files are checked in and trusted over comments), any
assertion failure raises and exits non-zero -- nothing here is a mere
comment.

Usage:  <python-with-mpmath> scripts/gen_trig_ulp_vectors.py
Writes tests/trig_ulp_vectors.inc relative to the repo root.
"""

import math
import os
import random
import struct
import sys

import mpmath as mp

mp.mp.prec = 320
SEED = 20260820  # documented fixed seed
random.seed(SEED)

D_MAX = float(2**23)  # matches docs/NEON_TRIG_DERIVATION.md's D_max

D = struct.Struct("<d")
Q = struct.Struct("<Q")


def bits(x: float) -> int:
    return Q.unpack(D.pack(x))[0]


def from_bits(b: int) -> float:
    return D.unpack(Q.pack(b))[0]


def cr_cos(x: float) -> float:
    return float(mp.cos(mp.mpf(x)))


def cr_sin(x: float) -> float:
    if x == 0.0:
        # mpmath's float->mpf conversion drops the sign of zero (mpf(-0.0) ==
        # mpf(0.0)), so mp.sin loses it too. sin is odd: sin(-0)=-0 exactly.
        # Caught by the generator's own self-check below -- see house
        # doctrine on trusting the self-check over an unverified comment.
        return math.copysign(0.0, x)
    return float(mp.sin(mp.mpf(x)))


def vec(x: float) -> tuple:
    return (bits(x), bits(cr_cos(x)), bits(cr_sin(x)))


# ---------------------------------------------------------------------------
# Self-check: known closed-form values, verified before anything downstream
# trusts mpmath's output. Any failure here means the generator itself is
# broken, so it must never produce a checked-in reference set silently.
# ---------------------------------------------------------------------------


def _self_check() -> None:
    checks = [
        ("cos(0)", cr_cos(0.0), 1.0),
        ("sin(0)", cr_sin(0.0), 0.0),
        ("cos(pi)", cr_cos(math.pi), -1.0),
        ("sin(pi/2)", cr_sin(math.pi / 2), 1.0),
        ("cos(-pi)", cr_cos(-math.pi), -1.0),
        ("sin(-pi/2)", cr_sin(-math.pi / 2), -1.0),
    ]
    for name, got, want in checks:
        assert abs(got - want) <= 1e-15, (f"{name} self-check failed", got, want)
    # cos(pi/2) is not exactly 0 in double precision (pi/2 is not exact), but
    # must be tiny -- pin the sign and magnitude rather than an exact value.
    cpi2 = cr_cos(math.pi / 2)
    assert abs(cpi2) < 1e-15, ("cos(pi/2) self-check: not tiny", cpi2)
    # Cross-check a few points against Python's own libm as a sanity floor on
    # mpmath itself (double-rounded, so allow a couple ULP of slack).
    for x in (0.5, 1.0, 2.0, -3.0, 12345.6789):
        c_ref, s_ref = cr_cos(x), cr_sin(x)
        c_libm, s_libm = math.cos(x), math.sin(x)
        assert abs(c_ref - c_libm) <= 4 * abs(c_libm) * 2**-52 + 2**-1000, (
            "cos vs libm sanity check failed",
            x,
            c_ref,
            c_libm,
        )
        assert abs(s_ref - s_libm) <= 4 * abs(s_libm) * 2**-52 + 2**-1000, (
            "sin vs libm sanity check failed",
            x,
            s_ref,
            s_libm,
        )


_self_check()

# ---------------------------------------------------------------------------
# Main buckets
# ---------------------------------------------------------------------------

bucket_counts = []
pts = []

# uniform [-2pi, 2pi]
n_before = len(pts)
for _ in range(1500):
    pts.append(random.uniform(-2 * math.pi, 2 * math.pi))
bucket_counts.append(("uniform_2pi", len(pts) - n_before))

# uniform [-1e4, 1e4]
n_before = len(pts)
for _ in range(1000):
    pts.append(random.uniform(-1e4, 1e4))
bucket_counts.append(("uniform_1e4", len(pts) - n_before))

# log-uniform magnitude out to +/-2^23
n_before = len(pts)
for _ in range(1000):
    mag = 2.0 ** random.uniform(-3, 23)
    pts.append(mag if random.random() < 0.5 else -mag)
bucket_counts.append(("log_uniform_domain", len(pts) - n_before))

# near k*pi/2 stress walk: both odd and even k, nearest double plus a few
# nextafter neighbours either side. K_MAX kept slightly below the true
# floor(2^23 * 2/pi) so that a handful of nextafter steps outward cannot
# push a generated point past D_MAX.
K_MAX = int(mp.floor(mp.mpf(D_MAX) * 2 / mp.pi)) - 8
OFFSETS = (-2, -1, 0, 1, 2)
n_before = len(pts)
stress_target = 1500
attempts = 0
while len(pts) - n_before < stress_target and attempts < stress_target * 20:
    attempts += 1
    parity = attempts % 2  # alternate odd/even k so both are well represented
    k = random.randint(0, K_MAX) * 2 + parity
    if random.random() < 0.5:
        k = -k
    xk = mp.mpf(k) * mp.pi / 2
    xk_d = float(xk)
    for off in OFFSETS:
        if len(pts) - n_before >= stress_target:
            break
        xoff = xk_d
        for _ in range(abs(off)):
            xoff = math.nextafter(xoff, math.inf if off > 0 else -math.inf)
        if abs(xoff) > D_MAX:
            continue
        pts.append(xoff)
bucket_counts.append(("near_k_pi_2_stress", len(pts) - n_before))

# self-check: every main-bucket point must sit inside the vectorized domain
for x in pts:
    assert abs(x) <= D_MAX, ("main-bucket point outside D_MAX", x)

main_vecs = [vec(x) for x in pts]

# ---------------------------------------------------------------------------
# Specials (outside the main gate budget; scalar-libm fixup + edge cases)
# ---------------------------------------------------------------------------

nan_bits = bits(math.nan)
pinf = math.inf
ninf = -math.inf

# Index order is a contract with tests/test_trig_ulp_gates.cpp: +/-0 stay at
# indices 0-1 (asserted by position), and the three non-finites sit at 2-4 so
# that every tier -- including the 8-wide one, whose vector body covers lanes
# 0-7 of the 11 specials -- evaluates them INSIDE the SIMD kernel. At the tail
# they would be handed to the scalar libm fixup and the kernel's own
# NaN/inf handling would go untested.
specials_head = [0.0, -0.0]
specials_tail = [
    D_MAX,
    -D_MAX,
    math.nextafter(D_MAX, math.inf),
    2.0 * D_MAX,  # 2^24
    1e9,
    1e300,
]
specials = [vec(x) for x in specials_head]

# +/-inf and NaN: cos/sin are NaN. mpmath cannot evaluate these directly, so
# the reference bits are hardcoded to the NaN encoding.
for x in (pinf, ninf, math.nan):
    specials.append((bits(x), nan_bits, nan_bits))
specials += [vec(x) for x in specials_tail]
specials_labels = specials_head + [pinf, ninf, math.nan] + specials_tail

# self-check: NaN/Inf encodings are exactly what IEEE-754 double predicts
for (xb, cb, sb), xv in zip(specials[2:5], (pinf, ninf, math.nan)):
    assert math.isnan(from_bits(cb)) and math.isnan(from_bits(sb)), (
        "specials NaN/Inf reference must be NaN",
        xv,
    )
    if math.isinf(xv):
        assert math.isinf(from_bits(xb)) and math.copysign(1.0, from_bits(xb)) == math.copysign(
            1.0, xv
        ), ("specials Inf encoding mismatch", xv)
    else:
        assert math.isnan(from_bits(xb)), ("specials NaN encoding mismatch", xv)

# self-check: cos(+/-0)=1 and sin(+/-0)=+/-0 (sign preserved), the contract
# tests/test_trig_ulp_gates.cpp asserts explicitly for every tier.
assert from_bits(specials[0][1]) == 1.0, "cos(+0) reference must be exactly 1.0"
assert from_bits(specials[1][1]) == 1.0, "cos(-0) reference must be exactly 1.0"
assert from_bits(specials[0][2]) == 0.0 and math.copysign(1.0, from_bits(specials[0][2])) == 1.0, (
    "sin(+0) reference must be exactly +0.0"
)
assert from_bits(specials[1][2]) == 0.0 and math.copysign(1.0, from_bits(specials[1][2])) == -1.0, (
    "sin(-0) reference must be exactly -0.0"
)

# ---------------------------------------------------------------------------
# Emit
# ---------------------------------------------------------------------------

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out = os.path.join(root, "tests", "trig_ulp_vectors.inc")
with open(out, "w") as f:
    f.write("// Auto-generated correctly-rounded cos()/sin() reference vectors.\n")
    f.write("// {input_bits, cos_bits, sin_bits}; cos/sin evaluated at 320-bit precision\n")
    f.write("// (mpmath) then each rounded once to nearest double. Gates the x86\n")
    f.write("// vector_cos_<tier>/vector_sin_<tier> SIMD kernels (issue #95) at a\n")
    f.write("// per-tier ULP budget.\n")
    f.write(f"// Fixed seed {SEED}. DO NOT EDIT -- regenerate with\n")
    f.write("// scripts/gen_trig_ulp_vectors.py.\n")
    f.write(
        "struct TrigUlpVector { std::uint64_t x_bits; std::uint64_t cos_bits; "
        "std::uint64_t sin_bits; };\n"
    )
    f.write(f"static constexpr TrigUlpVector kTrigUlpVectors[{len(main_vecs)}] = {{\n")
    for xb, cb, sb in main_vecs:
        f.write(f"    {{0x{xb:016x}ULL, 0x{cb:016x}ULL, 0x{sb:016x}ULL}},\n")
    f.write("};\n\n")
    f.write("// Specials: outside the main gate budget. Beyond-D_MAX finite points\n")
    f.write("// exercise the batch wrappers' per-lane scalar-libm fixup path; +/-inf and\n")
    f.write("// NaN must produce NaN exactly.\n")
    f.write(f"static constexpr TrigUlpVector kTrigUlpSpecials[{len(specials)}] = {{\n")
    for xb, cb, sb in specials:
        f.write(f"    {{0x{xb:016x}ULL, 0x{cb:016x}ULL, 0x{sb:016x}ULL}},\n")
    f.write("};\n")

print(f"wrote {out}: {len(main_vecs)} main vectors, {len(specials)} specials")
print("bucket counts: " + ", ".join(f"{name}={count}" for name, count in bucket_counts))
sys.exit(0)

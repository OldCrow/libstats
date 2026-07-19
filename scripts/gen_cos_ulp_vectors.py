#!/usr/bin/env python3
"""Generate src/cos_ulp_vectors.inc -- correctly-rounded cos() reference set.

Backs tests/test_simd_neon_cos_accuracy.cpp (the <1 ULP regression gate for
the clean-room vector_cos_neon kernel). Each entry is (input_bits, cos_bits):
cos evaluated at 320-bit precision with mpmath, rounded once to nearest
double. Architecture-neutral (pure mathematics).

Buckets (all inside the kernel's supported domain |x| <= 2^23):
  - uniform in [-2pi, 2pi] (the common statistics range, e.g. von Mises)
  - uniform in [-1e4, 1e4]
  - log-ish uniform out to +/-2^23 (domain-wide coverage)
  - near odd multiples of pi/2 (the stress set: results near zero, where
    reduction error dominates and the pre-2026-07-19 kernel had sign errors)

Usage:  python3 -m venv .venv && .venv/bin/pip install mpmath
        .venv/bin/python scripts/gen_cos_ulp_vectors.py
Writes src/cos_ulp_vectors.inc relative to the repo root.
"""

import os
import random
import struct

import mpmath as mp

mp.mp.prec = 320
random.seed(20260719)

D = struct.Struct("<d")
Q = struct.Struct("<Q")


def bits(x: float) -> int:
    return Q.unpack(D.pack(x))[0]


def from_bits(b: int) -> float:
    return D.unpack(Q.pack(b))[0]


def cr_cos(x: float) -> float:
    return float(mp.cos(mp.mpf(x)))


pts = []
# uniform [-2pi, 2pi]
for _ in range(1500):
    pts.append(random.uniform(-2 * 3.141592653589793, 2 * 3.141592653589793))
# uniform [-1e4, 1e4]
for _ in range(1000):
    pts.append(random.uniform(-1e4, 1e4))
# domain-wide, log-ish uniform magnitude out to 2^23
for _ in range(1000):
    mag = 2.0 ** random.uniform(-3, 23)
    pts.append(mag if random.random() < 0.5 else -mag)
# stress: nearest doubles to odd k*pi/2 and close bit-neighbours
for _ in range(1500):
    k = random.randint(-2000000, 2000000) * 2 + 1
    xk = float(mp.mpf(k) * mp.pi / 2)
    if abs(xk) > 2.0**23:
        continue
    pts.append(from_bits(bits(xk) + random.randint(-4, 4)))

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out = os.path.join(root, "src", "cos_ulp_vectors.inc")
with open(out, "w") as f:
    f.write("// Auto-generated correctly-rounded cos() reference vectors.\n")
    f.write("// {input_bits, cos_bits}; cos evaluated at 320-bit precision (mpmath) then\n")
    f.write("// rounded once to nearest double. Used to gate the clean-room\n")
    f.write("// vector_cos_neon kernel at <1 ULP. Architecture-neutral (pure mathematics).\n")
    f.write("// DO NOT EDIT -- regenerate with scripts/gen_cos_ulp_vectors.py.\n")
    f.write("struct CosUlpVector { std::uint64_t x_bits; std::uint64_t cos_bits; };\n")
    f.write(f"static constexpr CosUlpVector kCosUlpVectors[{len(pts)}] = {{\n")
    for x in pts:
        f.write(f"    {{0x{bits(x):016x}ULL, 0x{bits(cr_cos(x)):016x}ULL}},\n")
    f.write("};\n")

print(f"wrote {out}: {len(pts)} vectors")

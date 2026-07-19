#!/usr/bin/env python3
"""Generate exp_ulp_vectors.inc — correctly-rounded exp() reference set.

Issue #33: high-precision, architecture-neutral reference for validating the
< 1 ULP accuracy gate of table-based exp kernels (originally the AVX-512 Stage 3
experiment; now also used by the NEON production regression test). Aligns with
issue #46 (mpmath as the arbitrary-precision reference).

Emits an array of {input, correctly_rounded_exp} pairs as bit patterns. The
reference is exp(x) evaluated at 200-bit precision then rounded once to the
nearest double, i.e. the correctly-rounded result. Inputs deliberately cover:
  - the dense normal range (uniform sweep in [-700, 700])
  - argument-reduction grid points (k*ln2/128 for k around 0)
  - overflow / underflow / subnormal-output edges (~709.78, ~-708, ~-745)
  - tiny |x| where exp(x) ~= 1 + x
  - exact zero

Special IEEE inputs (+/-inf, NaN) are validated separately in the harness, not
here, since "correctly-rounded" is not meaningful for them.

Usage:
    python scripts/gen_exp_ulp_vectors.py > src/exp_ulp_vectors.inc
"""

import struct
import sys

import mpmath


def as_uint64(d: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", d))[0]


def correctly_rounded_exp(x: float) -> float:
    # exp(x) at high precision, rounded once to nearest double.
    return float(mpmath.e ** mpmath.mpf(x))


def build_inputs():
    xs = []
    ln2 = mpmath.log(2)

    # Dense uniform sweep across the normal range.
    steps = 400
    lo, hi = -700.0, 700.0
    for i in range(steps + 1):
        xs.append(lo + (hi - lo) * i / steps)

    # Argument-reduction grid: near-multiples of ln2/128 stress table indexing.
    for k in range(-260, 261):
        xs.append(float(k * ln2 / 128))

    # Small |x| where the polynomial dominates and the tail matters most.
    for e in range(-45, 1):
        xs.append(2.0**e)
        xs.append(-(2.0**e))

    # Overflow / underflow / subnormal-output edges.
    edges = [
        0.0,
        709.782712893383973096,   # ~ largest x with finite exp
        709.7827128933841,        # just over -> overflow
        -708.396418532264,        # near smallest normal output
        -744.4400719213812,       # ~ smallest x with nonzero (subnormal) exp
        -745.1332191019412,       # underflows to 0
        1.0, -1.0, 0.5, -0.5,
    ]
    xs.extend(edges)

    # De-duplicate while keeping only finite, in-domain-ish values.
    seen = set()
    out = []
    for x in xs:
        b = as_uint64(x)
        if b in seen:
            continue
        seen.add(b)
        out.append(x)
    return out


def main():
    mpmath.mp.prec = 200
    xs = build_inputs()

    print("// Auto-generated correctly-rounded exp() reference vectors (Issue #33).")
    print("// {input_bits, exp_bits}; exp evaluated at 200-bit precision (mpmath) then")
    print("// rounded once to nearest double. Architecture-neutral -- used to measure the")
    print("// ULP error of table-based exp kernels (NEON production, AVX-512 experimental)")
    print("// against the < 1 ULP accuracy gate (see issue #46).")
    print("// DO NOT EDIT -- regenerate with scripts/gen_exp_ulp_vectors.py.")
    print("struct ExpUlpVector { std::uint64_t x_bits; std::uint64_t exp_bits; };")
    print(f"static constexpr ExpUlpVector kExpUlpVectors[{len(xs)}] = {{")
    for x in xs:
        y = correctly_rounded_exp(x)
        print(f"    {{0x{as_uint64(x):016x}ULL, 0x{as_uint64(y):016x}ULL}},")
    print("};")


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Generate log_ulp_vectors.inc -- correctly-rounded log() reference set.

Issue #33 Q1: high-precision reference for validating the < 1 ULP accuracy gate of
the NEON table-gather log kernel. Aligns with issue #46 (mpmath as the
arbitrary-precision reference). Architecture-neutral -- pure mathematics.

Emits an array of {input, correctly_rounded_log} pairs as bit patterns. The
reference is log(x) evaluated at 200-bit precision then rounded once to the
nearest double. Inputs deliberately cover:
  - a full mantissa sweep across [0.5, 2] (exercises all 128 table subintervals)
  - powers of two across the whole normal exponent range (exercises all k)
  - a dense near-1.0 band (the cancellation region ARM special-cases)
  - subnormal inputs (normalized path) and the domain edges

Special/invalid inputs (0, negative, +inf, NaN) are validated separately in the
harness, not here, since "correctly-rounded" is not meaningful for them.

Usage:
    python scripts/gen_log_ulp_vectors.py > src/log_ulp_vectors.inc
"""

import math
import struct
import sys

import mpmath

DBL_MIN = 2.2250738585072014e-308      # smallest positive normal
DBL_TRUE_MIN = 5e-324                  # smallest positive subnormal
DBL_MAX = 1.7976931348623157e308


def as_uint64(d: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", d))[0]


def correctly_rounded_log(x: float) -> float:
    return float(mpmath.log(mpmath.mpf(x)))


def build_inputs():
    xs = []

    # Mantissa sweeps: [1,2] and [0.5,1] exercise every table subinterval and both
    # signs of k, plus straddle the near-1 cancellation region.
    steps = 512
    for i in range(steps + 1):
        xs.append(1.0 + i / steps)          # [1, 2]
        xs.append(0.5 + 0.5 * i / steps)    # [0.5, 1]

    # Powers of two across the normal exponent range -> every k value.
    for e in range(-1021, 1024):
        xs.append(math.ldexp(1.0, e))

    # Odd mantissas across many binades so k and the table index vary together.
    for e in range(-60, 61):
        xs.append(math.ldexp(1.3, e))
        xs.append(math.ldexp(1.7, e))

    # Dense near-1.0 band (ARM special-cases |log(x)| < ~2^-4 for cancellation).
    for i in range(-256, 257):
        xs.append(1.0 + i * 1e-4)           # ~[0.974, 1.026]

    # Subnormal inputs (normalized internally) and domain edges.
    xs += [DBL_TRUE_MIN, 1e-320, 1e-315, DBL_MIN, DBL_MAX, 1.0, 2.0, 0.5,
           math.e, 10.0, 1e-300, 1e300]

    # De-duplicate by bit pattern; keep only finite, strictly positive values.
    seen = set()
    out = []
    for x in xs:
        if not (math.isfinite(x) and x > 0.0):
            continue
        b = as_uint64(x)
        if b in seen:
            continue
        seen.add(b)
        out.append(x)
    return out


def main():
    mpmath.mp.prec = 200
    xs = build_inputs()

    print("// Auto-generated correctly-rounded log() reference vectors (Issue #33 Q1).")
    print("// {input_bits, log_bits}; log evaluated at 200-bit precision (mpmath) then")
    print("// rounded once to nearest double. Used to measure the ULP error of the NEON")
    print("// table-gather log kernel against the < 1 ULP accuracy gate (see issue #46).")
    print("// Architecture-neutral (pure mathematics).")
    print("// DO NOT EDIT -- regenerate with scripts/gen_log_ulp_vectors.py.")
    print("struct LogUlpVector { std::uint64_t x_bits; std::uint64_t log_bits; };")
    print(f"static constexpr LogUlpVector kLogUlpVectors[{len(xs)}] = {{")
    for x in xs:
        y = correctly_rounded_log(x)
        print(f"    {{0x{as_uint64(x):016x}ULL, 0x{as_uint64(y):016x}ULL}},")
    print("};")


if __name__ == "__main__":
    sys.exit(main())

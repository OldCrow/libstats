#!/usr/bin/env python3
"""Generate neon_erf_data.inc — lookup table for vector_erf_neon.

Produces 769 entries of (erf(k/128), 2/sqrt(pi)*exp(-(k/128)^2)) for k=0..768,
covering |x| in [0, 6-1/128] in steps of 1/128.  Entry 768 is the saturation
clamp entry used for |x| > 5.9921875 and NaN inputs.

Usage:
    python3 scripts/gen_neon_erf_table.py > src/neon_erf_data.inc
"""

import math
import sys

TWO_OVER_SQRTPI = 2.0 / math.sqrt(math.pi)
N_ENTRIES = 769  # k = 0..768, r = 0/128..768/128 = 0..6
STEP = 128       # 1/step = grid spacing = 1/128


def main():
    print("// Auto-generated erf table for vector_erf_neon.")
    print(f"// erf(k/{STEP}) and scale(k) = (2/sqrt(pi))*exp(-(k/{STEP})^2), k=0..{N_ENTRIES-1}.")
    print(f"// {N_ENTRIES} entries = {N_ENTRIES * 16:,} bytes.  Covers |x| in [0, 6-1/{STEP}]; entry")
    print(f"// {N_ENTRIES-1} is the saturation clamp (erf=1, scale≈0) for |x|>5.9921875 and NaN.")
    print("// DO NOT EDIT — regenerate with scripts/gen_neon_erf_table.py.")
    print("alignas(16) static constexpr struct {")
    print("    double erf_r;")
    print("    double scale;")
    print(f"}} kErfNeonTable[{N_ENTRIES}] = {{")
    for k in range(N_ENTRIES):
        r = k / STEP
        erf_r = math.erf(r)
        scale = TWO_OVER_SQRTPI * math.exp(-r * r)
        print(f"    {{{erf_r.hex()}, {scale.hex()}}},  // k={k}")
    print("};")


if __name__ == "__main__":
    main()

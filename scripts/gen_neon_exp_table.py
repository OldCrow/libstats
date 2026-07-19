#!/usr/bin/env python3
"""Generate neon_exp_data.inc -- lookup table for the NEON table-gather exp prototype.

Issue #33 Q1: table-based exp for NEON via the software-gather pattern (2x vld1q +
vuzp deinterleave), mirroring vector_erf_neon / neon_erf_data.inc.

Reproduces the SAME N=128 tail-corrected table as the (closed) AVX-512 experiment
(src/avx512_exp_data.inc, from ARM optimized-routines math/exp_data.c, SPDX: MIT OR
Apache-2.0 WITH LLVM-exception):

    2^(k/128) ~= H[k] * (1 + T[k])   for integer k in [0, 128)

The ONLY difference from gen_avx512_exp_table.py is the emitted C++ layout. AVX-512
has a hardware gather, so it uses two Structure-of-Arrays tables (one per gather).
NEON has NO hardware gather; the idiomatic software gather is a single 128-bit
vld1q_u64 per lane plus a vuzp deinterleave (exactly as kErfNeonTable does). So the
two values are stored interleaved as an Array-of-Structs of adjacent uint64 pairs,
matching ARM's own tab[] ordering (tab[2k]=tail, tab[2k+1]=sbits):

    kExpNeonTable[k] = { tail_bits, sbits }
        tail_bits = asuint64(T[k])                  (residual tail, reinterpreted to double at runtime)
        sbits     = asuint64(H[k]) - (k << 45)      (scale base; runtime adds (ki << 45))

The (52 - 7) = 45 shift is EXP_TABLE_BITS = 7 for N = 128.

Values are regenerated here from high precision (mpmath) rather than copied, then
asserted bit-exact against a sample of ARM's published tab[] entries so a
transcription or precision error is caught at generation time.

Usage:
    python scripts/gen_neon_exp_table.py > src/neon_exp_data.inc
"""

import struct
import sys

import mpmath

N = 128
SHIFT_BITS = 52 - 7  # 45; EXP_TABLE_BITS = 7 for N = 128
MASK64 = (1 << 64) - 1

# ARM optimized-routines math/exp_data.c, __exp_data.tab, N == 128 block.
# {tail_bits (tab[2k]), sbits (tab[2k+1])} for a sample of k. Used only as a
# generation-time cross-check; not the source of the emitted values. Identical
# reference set to gen_avx512_exp_table.py.
ARM_REF = {
    0: (0x0, 0x3FF0000000000000),
    1: (0x3C9B3B4F1A88BF6E, 0x3FEFF63DA9FB3335),
    2: (0xBC7160139CD8DC5D, 0x3FEFEC9A3E778061),
    3: (0xBC905E7A108766D1, 0x3FEFE315E86E7F85),
    4: (0x3C8CD2523567F613, 0x3FEFD9B0D3158574),
    5: (0xBC8BCE8023F98EFA, 0x3FEFD06B29DDF6DE),
    6: (0x3C60F74E61E6C861, 0x3FEFC74518759BC8),
    7: (0x3C90A3E45B33D399, 0x3FEFBE3ECAC6F383),
    8: (0x3C979AA65D837B6D, 0x3FEFB5586CF9890F),
    16: (0xBC801B15EAA59348, 0x3FEF72B83C7D517B),
    31: (0x3C6E149289CECB8F, 0x3FEF0CAFA93E2F56),
    32: (0x3C834D754DB0ABB6, 0x3FEF06FE0A31B715),
    33: (0x3C864201E2AC744C, 0x3FEF0170FC4CD831),
}


def as_uint64(d: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", d))[0]


def gen():
    mpmath.mp.prec = 200  # ample headroom over double's 53 bits
    entries = []
    for k in range(N):
        # 2^(k/128) in high precision.
        exact = mpmath.power(2, mpmath.mpf(k) / N)
        # H[k] = 2^(k/128) rounded to nearest double.
        h = float(exact)
        h_bits = as_uint64(h)
        # T[k] = 2^(k/128)/H[k] - 1 rounded to nearest double (the residual tail).
        tail = float(exact / mpmath.mpf(h) - 1)
        tail_bits = as_uint64(tail)
        # sbits base with the (k << 45) exponent contribution pre-subtracted.
        sb = (h_bits - (k << SHIFT_BITS)) & MASK64

        # Cross-check against ARM's published values where available.
        if k in ARM_REF:
            ref_tail, ref_sbits = ARM_REF[k]
            assert sb == ref_sbits, f"sbits mismatch k={k}: {sb:#018x} != {ref_sbits:#018x}"
            assert tail_bits == ref_tail, (
                f"tail mismatch k={k}: {tail_bits:#018x} != {ref_tail:#018x}"
            )

        entries.append((tail_bits, sb))
    return entries


def main():
    entries = gen()

    print("// Auto-generated exp lookup table for the NEON table-gather exp prototype (Issue #33 Q1).")
    print("// 2^(k/128) ~= H[k]*(1 + T[k]) for k = 0..127; stored Array-of-Structs as adjacent")
    print("// uint64 pairs so a single 128-bit vld1q_u64 pulls both values per lane and a vuzp")
    print("// deinterleaves them (the NEON software-gather pattern, cf. kErfNeonTable). This is")
    print("// the same table as src/avx512_exp_data.inc, only the memory layout differs.")
    print("//   tail_bits = asuint64(T[k])                (residual tail, reinterpreted to f64)")
    print("//   sbits     = asuint64(H[k]) - (k << 45)    (scale base; runtime adds ki << 45)")
    print("// Reproduces ARM optimized-routines math/exp_data.c (N=128),")
    print("// SPDX: MIT OR Apache-2.0 WITH LLVM-exception. See THIRD_PARTY_NOTICES.md.")
    print("// DO NOT EDIT -- regenerate with scripts/gen_neon_exp_table.py.")
    print("alignas(16) static constexpr struct {")
    print("    std::uint64_t tail_bits;  // asuint64(T[k])")
    print("    std::uint64_t sbits;      // asuint64(H[k]) - (k << 45)")
    print(f"}} kExpNeonTable[{N}] = {{")
    for k, (tail_bits, sb) in enumerate(entries):
        print(f"    {{0x{tail_bits:016x}ULL, 0x{sb:016x}ULL}},  // k={k}")
    print("};")


if __name__ == "__main__":
    sys.exit(main())

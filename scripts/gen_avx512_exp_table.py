#!/usr/bin/env python3
"""Generate avx512_exp_data.inc — lookup tables for vector_exp_avx512_gather.

Issue #33 Stage 3: table-based exp for AVX-512 using hardware gather.

Reproduces the N=128 table from ARM optimized-routines math/exp_data.c
(SPDX: MIT OR Apache-2.0 WITH LLVM-exception), split into two Structure-of-Arrays
tables so each can be pulled with a single 8-wide gather:

    2^(k/128) ~= H[k] * (1 + T[k])   for integer k in [0, 128)

    kExpSbitsAvx512[k] = asuint64(H[k]) - (k << 45)   (uint64, the "sbits" base)
    kExpTailAvx512[k]  = T[k]                          (double,  the "tail" residual)

At runtime the kernel forms scale = asdouble(sbits + (ki << 45)) = 2^(ki/128),
and applies the tail as a polynomial-additive correction to reach < 1 ULP.

The (52 - 7) = 45 shift is EXP_TABLE_BITS = 7 for N = 128. This mirrors ARM's
`tab[2k+1] = asuint64(H[k]) - (k << 52)/N`.

Values are regenerated here from high precision (mpmath) rather than copied, then
asserted bit-exact against a sample of ARM's published tab[] entries so a
transcription or precision error is caught at generation time.

Usage:
    python scripts/gen_avx512_exp_table.py > src/avx512_exp_data.inc
"""

import struct
import sys

import mpmath

N = 128
SHIFT_BITS = 52 - 7  # 45; EXP_TABLE_BITS = 7 for N = 128
MASK64 = (1 << 64) - 1

# ARM optimized-routines math/exp_data.c, __exp_data.tab, N == 128 block.
# {tail_bits (tab[2k]), sbits (tab[2k+1])} for k = 0..31. Used only as a
# generation-time cross-check; not the source of the emitted values.
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
    9: (0x3C8EB51A92FDEFFC, 0x3FEFAC922B7247F7),
    10: (0x3C3EBE3D702F9CD1, 0x3FEFA3EC32D3D1A2),
    11: (0xBC6A033489906E0B, 0x3FEF9B66AFFED31B),
    12: (0xBC9556522A2FBD0E, 0x3FEF9301D0125B51),
    13: (0xBC5080EF8C4EEA55, 0x3FEF8ABDC06C31CC),
    14: (0xBC91C923B9D5F416, 0x3FEF829AAEA92DE0),
    15: (0x3C80D3E3E95C55AF, 0x3FEF7A98C8A58E51),
    16: (0xBC801B15EAA59348, 0x3FEF72B83C7D517B),
    31: (0x3C6E149289CECB8F, 0x3FEF0CAFA93E2F56),
    32: (0x3C834D754DB0ABB6, 0x3FEF06FE0A31B715),
    33: (0x3C864201E2AC744C, 0x3FEF0170FC4CD831),
}


def as_uint64(d: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", d))[0]


def as_double_hex(d: float) -> str:
    return float.hex(d)


def gen():
    mpmath.mp.prec = 200  # ample headroom over double's 53 bits
    sbits = []
    tails = []
    for k in range(N):
        # 2^(k/128) in high precision.
        exact = mpmath.power(2, mpmath.mpf(k) / N)
        # H[k] = 2^(k/128) rounded to nearest double.
        h = float(exact)
        h_bits = as_uint64(h)
        # T[k] = 2^(k/128)/H[k] - 1 rounded to nearest double (the residual tail).
        tail = float(exact / mpmath.mpf(h) - 1)
        # sbits base with the (k << 45) exponent contribution pre-subtracted.
        sb = (h_bits - (k << SHIFT_BITS)) & MASK64

        # Cross-check against ARM's published values where available.
        if k in ARM_REF:
            ref_tail, ref_sbits = ARM_REF[k]
            assert sb == ref_sbits, f"sbits mismatch k={k}: {sb:#018x} != {ref_sbits:#018x}"
            assert as_uint64(tail) == ref_tail, (
                f"tail mismatch k={k}: {as_uint64(tail):#018x} != {ref_tail:#018x}"
            )

        sbits.append(sb)
        tails.append(tail)
    return sbits, tails


def main():
    sbits, tails = gen()

    print("// Auto-generated exp lookup tables for vector_exp_avx512_gather (Issue #33).")
    print("// 2^(k/128) ~= H[k]*(1 + T[k]) for k = 0..127; split Structure-of-Arrays so")
    print("// each table is pulled with a single 8-wide _mm512_i64gather.")
    print("//   kExpSbitsAvx512[k] = asuint64(H[k]) - (k << 45)  (scale base, uint64)")
    print("//   kExpTailAvx512[k]  = T[k]                         (tail residual, double)")
    print("// Table values reproduce ARM optimized-routines math/exp_data.c (N=128),")
    print("// SPDX: MIT OR Apache-2.0 WITH LLVM-exception. See THIRD_PARTY_NOTICES.md.")
    print("// DO NOT EDIT -- regenerate with scripts/gen_avx512_exp_table.py.")
    print(f"alignas(64) static constexpr std::uint64_t kExpSbitsAvx512[{N}] = {{")
    for k in range(0, N, 4):
        row = ", ".join(f"0x{sbits[j]:016x}ULL" for j in range(k, k + 4))
        print(f"    {row},")
    print("};")
    print(f"alignas(64) static constexpr double kExpTailAvx512[{N}] = {{")
    for k in range(0, N, 4):
        row = ", ".join(as_double_hex(tails[j]) for j in range(k, k + 4))
        print(f"    {row},")
    print("};")


if __name__ == "__main__":
    sys.exit(main())

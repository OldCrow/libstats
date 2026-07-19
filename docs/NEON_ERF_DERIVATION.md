# Derivation — double-precision vector erf() for AArch64 NEON

> Note: this is the original clean-room derivation, authored in an isolated workspace
> from a functional specification with no access to any existing erf implementation
> (the remediation of Issue #67). References to `SPEC.md` / `gen_table.py` are to that
> clean-room workspace; the in-tree table generator is `scripts/gen_neon_erf_table.py`,
> which re-derives and self-checks the coefficients shown here. The point-by-point
> comparison confirming no shared expression with the superseded code is in
> `docs/NEON_ERF_DIVERGENCE_AUDIT.md`.

Date: 2026-07-19
Author: clean-room implementation, derived entirely from the mathematics in
SPEC.md. No existing implementation of erf (or any related special function)
was consulted, searched for, or recalled. Only general calculus (Taylor
series, Hermite polynomials, Sterbenz's lemma, IEEE-754 rounding) is cited.
mpmath is used strictly as a black-box high-precision oracle for *values* of
erf and exp; sympy is used to double-check my own symbolic work.

## 1. The local series and its coefficients

Start from the definition: `erf'(x) = (2/√π) e^(−x²)`. Let `f(x) = e^(−x²)`,
so `f'(x) = −2x f(x)`.

Claim: `f⁽ⁿ⁾(x) = (−1)ⁿ Hₙ(x) e^(−x²)` where `Hₙ` are the physicists'
Hermite polynomials defined by the recurrence

    H₀ = 1,  H₁ = 2x,  H_{n+1}(x) = 2x Hₙ(x) − 2n H_{n−1}(x),

with the standard derivative identity `Hₙ'(x) = 2n H_{n−1}(x)`.

Proof by induction: the base case n = 0 is `f = H₀ f`. For the step,

    d/dx [(−1)ⁿ Hₙ f] = (−1)ⁿ (Hₙ' − 2x Hₙ) f
                      = (−1)^{n+1} (2x Hₙ − 2n H_{n−1}) f
                      = (−1)^{n+1} H_{n+1} f.     ∎

Hence for n ≥ 1:

    erf⁽ⁿ⁾(x) = (2/√π) (−1)^{n−1} H_{n−1}(x) e^(−x²).

Taylor expansion about a grid point `r`, with residual `d = a − r`:

    erf(a) = erf(r) + Σ_{k≥1} erf⁽ᵏ⁾(r)/k! · dᵏ
           = E + S · ( d + c₂(r) d² + c₃(r) d³ + … ),

    E = erf(r),   S = erf'(r) = (2/√π) e^(−r²),
    c_k(r) = (−1)^{k−1} H_{k−1}(r) / k!,   c₁ = 1.

Working out the recurrence gives the explicit coefficients used in the
kernel (w = r²):

    c₂ = −r
    c₃ = (2w − 1)/3
    c₄ = r(3 − 2w)/6
    c₅ = (4w² − 12w + 3)/30
    c₆ = −r(4w² − 20w + 15)/90
    c₇ = (8w³ − 60w² + 90w − 15)/630
    c₈ = −r(8w³ − 84w² + 210w − 105)/2520

Verification (`gen_table.py`, sections 1–2 of its output):
- symbolic: each c_k above matches the k-th Taylor coefficient of
  `erf(r+d)` computed independently by `sympy.series`, divided by S —
  `simplify(difference) == 0` for k = 1..12. PASS.
- numeric: summing the series to k = 12 at 40 random (r, d) pairs,
  r ∈ [0,6], |d| ≤ 0.004, reproduces mpmath's 50-digit erf to better than
  10⁻²⁴ relative. PASS.

At r = 0 the even coefficients vanish (H_{2m+1}(0) = 0), which is exactly the
odd Maclaurin series `erf(a) = (2/√π)(a − a³/3 + a⁵/10 − a⁷/42 + …)`. This
matters for the error analysis below.

## 2. Saturation and edge cases

`erf(a) → 1` and binary64 has a half-ULP gap of 2⁻⁵⁴ below 1.0, so beyond
some threshold every erf(a) rounds to exactly 1.0. Bisecting on the bit
pattern against the high-precision oracle (`gen_table.py` section 3) gives

    A_max = 5.921587195794507   (bits 0x4017afb48dc96627)

as the *smallest* double whose erf rounds to 1.0. I use exactly this value
as the saturation bound: for |x| ≥ A_max return copysign(1, x). This is
correctly rounded *by construction* for the whole saturated range, and it
removes a genuine knife-edge: erf(A_max) sits within ~10⁻¹⁹ of the rounding
midpoint between 1 and 1−2⁻⁵³, and during development the series path
(h = 2⁻⁵, N = 8) computed it a hair below the midpoint — a 1-ULP miss that no
reasonable series accuracy can guarantee to avoid. Saturating exactly at the
threshold makes the question moot. (First measured with A_max = 6.0; the
final A_max = 5.921587195794507 is why every config now passes the edge
tests.)

Edge behaviour falls out of the vector code without branches:
- ±0: cell 0 has E = 0, r = 0, so the result is S·(±0 + 0) = ±0 with the
  sign restored from x (sign bit copy keeps −0).
- ±inf: |x| clamps to A_max, saturation compare (|x| ≥ A_max) is true → ±1.
- NaN: fmin propagates NaN; fcvtns(NaN) = 0 is a safe table index; d = NaN
  then poisons the series; the saturation compare is false for NaN; the sign
  copy leaves a NaN. Result: NaN.
- Tiny/subnormal a: j = 0, d = a, p = fma(d², t, d) rounds to d (the fused
  multiply-add does not lose the underflowing d²·t term separately), so the
  result is S·a — the correct leading behaviour (2/√π)a — with no spurious
  under/overflow. Verified for 5e-324.
- Oddness is exact: the kernel computes with |x| and copies x's sign bit.

## 3. Numerical structure (why each piece is exact or compensated)

- Grid spacing is a power of two, h = 2^−k, grid r_j = j·h, j ≤ 6·2^k.
  Then `a·(1/h)` is exact (pure exponent shift), so the index
  `j = round-to-nearest(a·2^k)` is exact, and `|d| ≤ h/2` always.
- `d = a − r` is exact: for j = 0 it is a itself; for j ≥ 1,
  a ∈ [r − h/2, r + h/2] ⊂ [r/2, 2r], so Sterbenz's lemma applies.
- Table entry per grid point: {E_hi, S, E_lo, r_j}, 32 bytes, 32-byte
  aligned. E_hi = RN(erf(r_j)), E_lo = RN(erf(r_j) − E_hi) gives erf(r_j) to
  ~2⁻¹⁰⁶; S = RN((2/√π) e^(−r_j²)); r_j stored so the kernel need not
  recompute it. Two 128-bit loads per lane + two unzips form the vector
  gather (NEON has no gather instruction).
- The c_k(r) are evaluated at run time as the short polynomials in r and
  w = r² above (a few FMAs). Their contribution to the result is at most
  ~|c₂|·d ≈ 2.3% of the leading d term, so ~1-ulp errors in the c_k are
  ~10⁻¹⁸ relative — storing them per grid point would triple the table for
  no measurable accuracy gain.
- Evaluation: Horner in d for t = c₂ + d(c₃ + …), then p = fma(d², t, d),
  then result = E_hi + fma(S, p, E_lo). The final add contributes ≤ 0.5 ulp;
  the compensated E and fused S·p keep everything before it well below that.
  The measured effect of compensation (comp vs nocomp columns below):
  max error drops from 1–3 ulp to 1 ulp and mean drops ~8×.

## 4. Truncation error: where it binds and why odd N is efficient

The first neglected term is ≈ S·c_{N+1}(r)·d^{N+1}, |d| ≤ h/2, and it must be
small *relative to erf(a)*. For moderate and large r the factor e^(−r²) in S
kills it. The binding region is the first few cells, where erf(a) ≈ (2/√π)a
is itself small:
- cell 0 (r = 0): relative error ≈ |c_{N+1}(0)|·(h/2)^N. Even-index c's
  vanish at r = 0 (H_{odd}(0) = 0), so if N+1 is even this term disappears.
- cell 1 (r = h, a as small as h/2): for even N+1 the leading factor is
  c_{N+1}(h) ≈ c_{N+1}'(0)·h, which contributes an extra factor of h.

So stopping after an odd power d^N (N odd, next coefficient c_{N+1} even-index
and hence O(h) near 0) buys roughly an extra factor of h in the binding
region. The predicted worst cases (gen_table.py section 4, units of 2⁻⁵²):

    h=2^-5 N=7: 1.26        h=2^-5 N=8: 0.074
    h=2^-6 N=6: 24.4        h=2^-6 N=7: 0.0049
    h=2^-7 N=5: 4.95        h=2^-7 N=6: 0.38
    h=2^-8 N=4: 6.6e3       h=2^-8 N=5: 0.077      h=2^-8 N=6: 0.0060

These predictions match the measured max-ULP table below (configs predicted
>1·2⁻⁵² show >1 ulp max; configs predicted ≪1 all achieve max 1 ulp, the
floor set by final rounding).

## 5. Measured design space (Apple M1, clang -O3, 22008-point
correctly-rounded reference set; hot = 2048 doubles, stream = 262144)

    config       comp maxULP  comp meanULP  nocomp maxULP  hot ns/el  stream ns/el  table
    h=2^-5 N=7   2.00         0.0507        3.00           1.84       1.90           6.2 KB
    h=2^-5 N=8   1.00         0.0306        1.00           2.30       2.22           6.2 KB
    h=2^-6 N=6   28.00        0.1625        28.00          1.66       1.66          12.3 KB
    h=2^-6 N=7   1.00         0.0248        1.00           1.90       1.90          12.3 KB
    h=2^-7 N=5   6.00         0.1550        6.00           1.41       1.41          24.6 KB
    h=2^-7 N=6   1.00         0.0224        2.00           1.66       1.67          24.6 KB
    h=2^-7 N=7   1.00         0.0219        1.00           1.89       1.89          24.6 KB
    h=2^-8 N=4   7810.00      19.09         7810.00        1.23       1.24          49.2 KB
    h=2^-8 N=5   1.00         0.0218        1.00           1.40       1.37          49.2 KB
    h=2^-8 N=6   1.00         0.0200        1.00           1.66       1.66          49.2 KB
    scalar erf   3.00         0.4017        —              10.90      10.97         —

Unroll and index-method study on the max-1-ulp configs (hot / stream
ns/el): unroll ×1 vs ×2 vs ×4 differ by ≤ 0.05 ns (noise level; ×2
marginally best most runs). The "add 1.5·2⁵²" magic-number index extraction
is consistently ~0.01–0.15 ns *slower* than fcvtns (round-to-nearest
convert), because on AArch64 the convert is a single instruction and the
magic path needs an extra NaN-guarding select to keep the table index safe;
its accuracy is identical (max 1 ulp).

Worst-case inputs for all max-1-ulp configs sit at small |x| (0.01–0.36),
consistent with §4: the residual max error is the unavoidable ~0.5-ulp final
rounding plus sub-0.5-ulp accumulation in S·p, not truncation.

## 6. Final operating point

    h = 2^-8 (spacing 1/256), 1537-entry table of 32 B = 49.2 KB
    N = 5 series terms (d, c2 d^2, …, c5 d^5), coefficients from r at run time
    compensated E (E_hi + E_lo)
    index extraction: fmul by 2^8 + fcvtns (round-to-nearest convert)
    unroll: 2 vectors (4 doubles) per loop iteration
    A_max = 5.921587195794507 (exact rounds-to-1 threshold)

Rationale: among all configs achieving max 1 ulp / mean ≈ 0.02 ulp, it is
the fastest (1.40 hot / 1.37 stream ns per element, ~7.8× the system scalar
erf, which itself shows max 3 ulp on the same reference set). The shortest
series wins because the kernel is arithmetic-bound, and even the 49 KB
table stays effectively resident (hot and stream throughput are equal). If
table footprint mattered more than ~0.26 ns/element, h=2^-7 N=6 (24.6 KB)
or h=2^-6 N=7 (12.3 KB) achieve the same max 1 ulp; that trade-off is noted
in the code.

## 7. Reproducing

See README.md. All numbers above are from a real run on this machine
(Apple M1, AppleClang, -std=c++20 -O3).

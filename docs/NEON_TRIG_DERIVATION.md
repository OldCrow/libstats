# Derivation — double-precision NEON `sin(x)` / `cos(x)` (clean-room)

> **Workspace-name mapping**: `SPEC.md` and `gen_coeffs.py` are clean-room
> workspace files; the in-tree regenerators are
> `scripts/gen_neon_trig_cleanroom_table.py` (NEON) and
> `scripts/gen_trig_cleanroom_table.py` (x86 mirror).

Date: 2026-07-19.
Author: Claude (Fable 5), working solely from `SPEC.md` and the mathematics below.
Everything here — algorithm structure, split constants, polynomial
coefficients, error analysis — was derived in this session from first
principles; no existing implementation of any trigonometric or
argument-reduction routine was consulted, in any form. General mathematics
used: Taylor series with Lagrange remainder, Chebyshev near-minimax
approximation (via mpmath's generic `chebyfit`), the angle-addition
identities, Sterbenz's lemma, and standard IEEE-754 rounding error analysis.

Provenance note for the audit: partway through this session an unrelated
concurrent process moved this attempt's files into a quarantine folder and a
*second, independent* attempt at the same task began writing into the
directory root. While diagnosing the collision, a `diff` unavoidably showed
this author a few headline choices of that other attempt (its D_max = 2^20,
32-significant-bit split parts named `TRIG_PI2_A..D`, and use of chebyfit).
Every corresponding choice in this attempt (D_max = 2^23, 30-bit parts,
2/3/4-part exploration, chebyfit) was derived and written to disk *before*
that exposure (file mtimes: this attempt's `trig_coeffs.inc` 01:45, the other
attempt's files 01:55+), so no information flowed into this work; the two
attempts' constants and structures differ throughout.

---

## 1. Structure: quadrant reduction + parity-respecting cores

Write `x = n·(π/2) + r` with `n = round(x·2/π)` (round to nearest), so
`|r| ≤ π/4` (up to a hair more when the computed `n` differs from the ideal
one; see §3.5). Only `q = n mod 4` matters for recombination, because
`sin`/`cos` have period `2π = 4·(π/2)`.

### 1.1 Quadrant table (from the angle-addition identities)

With `b = q·π/2` and `sin b, cos b ∈ {0, ±1}`:

    sin(r + b) = sin r · cos b + cos r · sin b
    cos(r + b) = cos r · cos b − sin r · sin b

| q | (sin b, cos b) | sin(x)  | cos(x)  |
|---|----------------|---------|---------|
| 0 | (0, 1)         |  sin r  |  cos r  |
| 1 | (1, 0)         |  cos r  | −sin r  |
| 2 | (0, −1)        | −sin r  | −cos r  |
| 3 | (−1, 0)        | −cos r  |  sin r  |

Bit logic (branch-free), with `b0 = q & 1`, `b1 = (q >> 1) & 1`:

* **sin**: use the cos core iff `b0`; negate iff `b1`.
* **cos**: use the sin core iff `b0`; negate iff `b1 XOR b0`.
  (Check: q=0→+cos, q=1→−sin, q=2→−cos, q=3→+sin. ✓)

For negative `n`, the two low bits of the two's-complement integer equal
`n mod 4` (the non-negative residue), so the same bit logic applies; the
int64 conversion of the already-integral `n` is exact for `|n| < 2^63`.

The negations are implemented as an XOR of the IEEE sign bit
(`b1 << 63`, `(b1^b0) << 63`), which is exact for every value including
zeros, subnormals and NaN.

### 1.2 Exact symmetry

Mapping `x → −x` gives `n → −n` (round-to-nearest-even is symmetric),
`r → −r` and `rlo → −rlo` exactly (every operation in the reduction is a
multiply/FMS by sign-symmetric values), and `q → (−q) mod 4`. Applying the
table: each path maps sin output to its exact negation (the sin core is an
odd polynomial in `r`, evaluated on `u = r²` which is sign-invariant, times
`r`) and cos output to itself (even core). Hence `sin` is bitwise odd and
`cos` bitwise even — verified over 100 000 random points in the harness.

---

## 2. Split-constant construction

**Lemma (exact product).** If `p = m·2^g` with integer `|m| < 2^s`, and `n`
is an integer with `|n| < 2^(53−s)`, then `n·p = (n·m)·2^g` with
`|n·m| < 2^53`, hence exactly representable as a double, and any FMA/FMS
involving it commits no product rounding beyond its single final rounding.

**Construction** (in `gen_coeffs.py`, at 320-bit precision): with `R₀ = π/2`,
repeatedly round `Rᵢ` to its leading `s = 30` significant bits to obtain part
`pᵢ₊₁` (a double whose mantissa has ≥ 53−30 = 23 trailing zero bits), and
recurse on the remainder `Rᵢ₊₁ = Rᵢ − pᵢ₊₁`. Because the rounding is
to-nearest, remainders (and hence later parts) may be negative; that is
harmless — products stay exact. The final part is the full-precision double
of the last remainder. The generated parts (4-part config):

    p1 = 0x1.921fb54800000p+0     (30 sig bits)
    p2 = -0x1.de973dc800000p-31   (30 sig bits)
    p3 = -0x1.9d9cceb800000p-62   (30 sig bits)
    p4 = -0x1.1fc8f8cbb5bf7p-93   (53 sig bits, full tail)

Measured truncation of the whole split: `δ₄ = |π/2 − Σpₖ| ≈ 2^−147.2`
(likewise `δ₃ ≈ 2^−117.3`, `δ₂ ≈ 2^−85.7` for the 3- and 2-part configs).

With `s = 30`, products `n·pₖ` are exact for `|n| < 2^23`.

**Supported domain.** `D_max = 2^23` gives
`n_max = round(2^23·2/π) = 5 340 354 < 2^23`. ✓
(This exceeds the spec's 2^20 aim; the same construction extends further by
lowering `s`, at the price of more parts.)

---

## 3. Reduction error analysis

Computed sequence (per lane), `nf = round-to-nearest(x·2/π)` via `vrndnq`:

    r1 = fms(x,  nf, p1)
    r2 = fms(r1, nf, p2)      e2 = fms(r1 − r2, nf, p2)
    r3 = fms(r2, nf, p3)      e3 = fms(r2 − r3, nf, p3)
    r4 = fms(r3, nf, p4)      e4 = fms(r3 − r4, nf, p4)
    r = r4,  rlo = e2 + e3 + e4         (compensated configuration)

### 3.1 Step 1 is always exact

`n·p1` is exact and representable (lemma). For `n ≠ 0`,
`x = n·π/2 + r` with `|r| ≤ π/4`, so `x/(n·p1) ∈ [1 − π/(4·|n|·π/2), 1 + …]
⊆ [1/2, 2]`; by Sterbenz's lemma the difference `x − n·p1` is exactly
representable, and the FMS returns it exactly. For `n = 0`, `r1 = x`. So no
residual term is needed for step 1.

### 3.2 Middle/final steps: rounding vs. cancellation dichotomy

For step `k ≥ 2`, the subtrahend `n·pₖ` is exact (parts 2..3) or a fused
product (part 4). Two regimes:

* **Large cancellation** (`r_{k−1} ≈ n·pₖ`, i.e. the true remainder is much
  smaller than both): the operands are within a factor 2, Sterbenz applies,
  the step is **exact**, and the residual `eₖ ≈ 0`.
* **Small cancellation** (`|r| ≫ |n·pₖ|`): the step rounds by up to
  `ulp(r)/2`, but then `r_{k−1} − rₖ` **is** exact (the two are within a
  factor 2 of each other), so `eₖ` recovers the committed rounding error to
  full precision (its own rounding is second-order, ~2^−107).

Either way the pair `(r, rlo)` represents `x − n·Σpₖ` to far better than
double precision. Uncompensated, each of steps 2..4 can contribute
`ulp(r)/2`, i.e. up to ~1.5 ulp before the polynomial even runs — exactly
the ~1.9 ulp uniform max the harness measured for `COMP_R=0` (§6).

### 3.3 Truncation term and the stress bound

The remaining error is `|n|·δ₄ ≤ 2^22.4 · 2^−147.2 = 2^−124.8` (absolute).
The stress reference set (nearest doubles to `k·π/2` and ±1..3-ulp
neighbours, 2000 values of `k` across the domain) contains a worst observed
`|r|_min = 2^−60.49` (printed by the generator). Near such points
`sin ≈ r`, so one ULP of the result is `≈ 2^−60.49−52 = 2^−113`; the
truncation term is a factor ~2^12 below it — negligible. The dominant stress
error is the single final rounding of `r4` (≤ ulp(r)/2), which the
compensation converts into `rlo` and the sin core folds back in (§4.3),
leaving the measured **0.499/0.500 ULP** stress maxima — the correct-rounding
limit of a one-rounding pipeline.

For the 3-part split, `|n|·δ₃` can reach `2^−95`, i.e. ~2^18 ulp at
`|r|_min` — measured as >1 ULP only because the sampled minimal-`|r|` points
happen to have small `n`; the 2-part split fails catastrophically (~10^10
ULP), both as predicted. See the design table in §6.

### 3.4 Why `n` from a plain double product is enough

`fl(x·(2/π))` has relative error ≤ 2^−53, so the computed `nf` can differ
from the ideal `round(x·2/π)` only when the exact product lies within
`≈ 2^−30` of a half-integer (at `|x| ≤ 2^23`); then either choice yields
`|r| ≤ π/4·(1 + ~2^−29)`. The polynomial fit interval is padded by a factor
`1 + 10^−6` to cover this (§4.1).

### 3.5 Tiny arguments and `−0`

For `n = 0` the reduction returns `r = x` exactly — except `x = −0`, where
`fl(−0 − (−0·p1)) = +0` by the IEEE addition sign rules, which would break
`sin(−0) = −0`. The kernel therefore selects `sin(x) = x` for
`|x| < 2^−26` (lane mask, branch-free). This is also the correctly-rounded
result: `|sin x − x| = x³/6·(1 + O(x²)) ≤ |x|·2^−52/6 < |x|·2^−54.5`,
inside half an ulp. The cos path needs **no** shortcut: with `n = 0` its
split head `h = fl(1 − x²/2)` (§4.4) is already the correctly-rounded
cos for small `|x|` (and exactly 1 once `x²/2 < 2^−55`). An earlier draft
used a cos shortcut with threshold 2^−26 and the harness caught it: for
`x ∈ [2^−26.5, 2^−26)` the correct answer is `1 − 2^−53`, not 1 — a nice
demonstration that the reference harness bites.

---

## 4. Polynomial cores

### 4.1 Forms and fit

On `u = r² ∈ [0, U]`, `U = (π/4·(1+10^−6))² ≈ 0.61685`:

    sin(r) = r + r·(u·P̃(u)),   P̃(u) ≈ (sin√u − √u)/u^{3/2}
    cos(r) = 1 + u·Q̃(u),        Q̃(u) ≈ (cos√u − 1)/u

These parity-respecting forms make the leading terms (`r`, `1`) exact and
halve the operation count (Horner in `u`). Coefficients: `mpmath.chebyfit`
(near-minimax Chebyshev fit) at 320-bit precision on `[0, U]`, then rounded
once to double, then the **rounded** coefficient set is re-verified against
320-bit references on a 4001-point grid (worst relative error of the exact
mathematical value of the double-coefficient polynomial):

| degree of P̃/Q̃ | sin rel. err | cos rel. err |
|---------------|--------------|--------------|
| 5             | 2^−55.5      | 2^−51.6      |
| 6             | 2^−57.1      | 2^−60.6      |
| 7             | 2^−57.1      | 2^−59.5      |

Degree 6 is the plateau for sin (double coefficient rounding dominates
beyond it) and the sweet spot for cos; degree 5 is inadequate for cos
(2^−51.6 → ~1.9 ULP measured). **Chosen: P̃ and Q̃ both degree 6** (highest
terms `r^15` and `r^14`).

### 4.2 Taylor cross-check

Truncated Taylor with Lagrange remainder at `r = π/4` gives
`(π/4)^17/17! ≈ 4.6·10^−17 ≈ 2^−53.8·sin(π/4)` for sin through `r^15`, and
`(π/4)^16/16! ≈ 1.0·10^−15 ≈ 2^−49.3·cos(π/4)` for cos through `r^14` — so
plain Taylor at these degrees would *not* reach < 1 ULP for cos; the
equioscillating fit redistributes the error and buys the needed ~2^11. The
fitted coefficients converge to the Taylor values (−1/6, +1/120, …; −1/2,
+1/24, …) in their leading digits, as they must — a derivation sanity check.

### 4.3 sin evaluation order

    ps = Horner(P̃, u)                  (6 FMAs)
    s  = r + fma(r·u, ps, rlo)

The compensation term `rlo` enters where it is numerically tiny, so the only
full-ulp rounding is the final add. For tiny `r` the added term vanishes and
`s = r` exactly (with the §3.5 mask handling `−0`).

### 4.4 cos evaluation order: exact `1 − u/2` split

The fitted `Q̃` constant term is exactly `−1/2` (degree ≥ 6), so split:

    h  = fl(1 − u/2)          (u/2 exact; one rounding)
    hl = (1 − h) − u/2        (both subtractions exact: 1−h by Sterbenz
                               since h ∈ [0.69, 1]; the second by cancellation)
    cos ≈ h + (hl + u·(c0+½) + u²·Q̃'(u) − r·rlo)

with `Q̃'` the Horner sum of coefficients 1..6 and `c0+½ = 0` for the chosen
fit (kept symbolically so any degree works). `h + hl = 1 − u/2` **exactly**,
every correction is accumulated at ~2^−54 magnitude, and only the final add
pays a full rounding. The `−r·rlo` term is the first-order effect of the
compensated reduction on cos (`cos(r+rlo) ≈ cos r − r·rlo`, ≤ ~0.17 ulp,
2nd order ≤ 2^−107). This restructuring measurably cut the uniform cos max
from 0.98–1.7 to 0.78 ULP.

---

## 5. Special values and fallback

`|x| > D_max`, `±inf` and `NaN` all fail the ordered compare
`|x| ≤ D_max`, and those lanes are patched with the scalar library oracle
(`std::sin`/`std::cos`), which the harness verifies bitwise, including
`±inf → NaN`, `NaN → NaN`. `sin(±0) = ±0`, `cos(±0) = 1`, tiny/subnormal
`x`, and bitwise symmetry are all exercised by the edge tests (0 failures).

---

## 6. Design-space exploration (measured on this M1)

Uniform = 3500 points as in `gen_coeffs.py`; stress = 14 000 near-`k·π/2`
points; ULP vs correctly-rounded 320-bit references (with stored low-order
residual, so fractional ULP is real). ns/element, hot 2K / stream 256K
buffers; timings are best-of-3 and were noisy (±30%) while a concurrent
process shared the machine — accuracy numbers are exact and reproducible.

| parts | comp | degS/degC | uni sin/cos max (ULP) | stress sin/cos max (ULP) | ~hot sin (ns) |
|-------|------|-----------|------------------------|---------------------------|---------------|
| 2 | no  | 6/6 | 1.26 / 1.22 | 5.0·10^9 (fails)   | 2.0 |
| 3 | no  | 6/6 | 1.89 / 1.73 | 0.94 / 0.94        | 2.4 |
| 4 | no  | 6/6 | 1.89 / 1.73 | 1.00 / 1.00        | 2.5 |
| 2 | yes | 6/6 | 0.71 / 0.65 | 1.0·10^10 (fails)  | 2.2 |
| 3 | yes | 6/6 | 0.76 / 0.78 | 1.56 / 1.10        | 2.7 |
| **4** | **yes** | **6/6** | **0.76 / 0.78** | **0.50 / 0.50** | **2.7** |
| 4 | yes | 5/5 | 1.87 / 1.91 | 0.50 / 0.50        | 2.7 |
| 4 | yes | 5/6 | 0.78 / 0.78 | 0.50 / 0.50        | 2.8 |
| 4 | yes | 7/7 | 0.76 / 0.78 | 0.50 / 0.50        | 2.8 |
| 4 | yes | 6/6, unroll 1 | 0.76 / 0.78 | 0.50 / 0.50 | 2.6 |

**Chosen operating point: 4 parts, compensated r, degrees 6/6, unroll 2**
(unroll 1 vs 2 indistinguishable within noise on this out-of-order core;
2 kept as the default since it is never slower). The compensation costs
~0.2–0.4 ns/element and buys the stress set three-plus orders of magnitude
at D_max = 2^23 (and the uniform max down from 1.9 to 0.76). Stress maxima
are bucket-independent (same 0.499/0.500 for |x| ≤ 2^10, 2^20, 2^23),
confirming the |n|-independence predicted in §3.3.

Final accuracy (chosen config): uniform sin 0.763 max / 0.234 mean, cos
0.777 max / 0.230 mean; stress sin 0.499 max / 0.119 mean, cos 0.500 max /
0.132 mean; all edge tests exact.

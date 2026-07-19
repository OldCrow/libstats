# Derivation — clean-room NEON `log(x)` (binary64, AArch64)

Date: 2026-07-19.
Author: derived from first principles per SPEC.md §2. No existing
implementation of `log` (or any related routine) was consulted; every
constant below is produced by `gen_table.py` from the mathematics in this
document, and every bound is re-verified numerically by the generator or the
harness. Only general mathematics is cited (Taylor series, Chebyshev
interpolation, standard floating-point lemmas, IEEE-754 encoding).

## 1. IEEE-754 decomposition

A binary64 value has bit layout `[1 sign | 11 exponent E | 52 fraction F]`.
For a finite positive **normal** x:

    x = 2^(E-1023) * (1 + F * 2^-52) = 2^e * m,   e = E-1023,  m in [1, 2).

Bit manipulation, both derived directly from the layout:

* `e = (bits(x) >> 52) - 1023` (sign bit is 0 on this path, so no masking
  is needed before the shift);
* `m` is obtained by replacing the exponent field with that of 1.0
  (`0x3FF`): `bits(m) = (bits(x) & 0x000FFFFFFFFFFFFF) | 0x3FF0000000000000`.

Then `log x = e*ln2 + log m`. **Subnormals** (`E = 0`, `F != 0`,
`x = F * 2^-1074`): multiply by 2^64. The product is exact (scaling by a
power of two; no overflow since x < 2^-1022 implies x*2^64 < 2^-958, and the
smallest input 2^-1074 becomes 2^-1010 > 2^-1022, i.e. always normal). The
kernel then subtracts 64 from e. Resulting range: |e| <= 1074, and at most
1075 after the re-centering of §4 — this bound sizes the ln2 split in §5.

## 2. Grid and index extraction

Cover m in [1,2] with N+1 anchors

    r_j = 1 + j/N,   j = 0..N,   N = 2^K.

The index is the nearest anchor: j = round(N*(m-1)). Since
`N*(m-1) = F / 2^(52-K)` exactly, rounding to nearest (ties up) is the
integer computation

    j = (F + 2^(51-K)) >> (52-K).

Exactness: `F + 2^(51-K) < 2^52 + 2^51` never overflows 64 bits; the shift
implements floor((F + h)/2h * ... ) — precisely, with s = 2^(52-K),
`(F + s/2) >> (52-K) = floor((F + s/2)/s) = round(F/s)` (ties rounded up),
which is round(N*(m-1)). Hence

    |m - r_j| <= 1/(2N),    j in [0, N]

(j = N occurs iff F >= 2^52 - 2^(51-K), i.e. m within 1/(2N) of 2).
The tie direction is irrelevant: either neighbouring anchor satisfies the
same bound. Verified exhaustively per-cell by `gen_table.py`.

## 3. Anchor identity and the residual t (no division)

Store per anchor the **double** `R_j = fl(1/r_j)` (round-to-nearest). The
key point is to define the anchor logarithm against the *stored* value:

    L_j = -log(R_j)     (computed at 300-bit precision on the exact double R_j)

so that, with the *real-number* residual `t* = m*R_j - 1`,

    log m = -log(R_j) + log(m*R_j) = L_j + log(1 + t*)         (exact identity)

No approximation has been made yet; the rounding of 1/r_j merely perturbs
the anchor point, never the identity. The kernel computes

    t = fl(m*R_j - 1)   — one FMA (vfmaq_f64), single rounding, no division.

**Bound on t\*.** Write `m*R_j - 1 = (m - r_j)*R_j + (r_j*R_j - 1)`.
With |m - r_j| <= 1/(2N), R_j <= 1, and |r_j*R_j - 1| <= 2^-53 (R_j is a
half-ulp-accurate reciprocal, ulp(R_j) <= 2^-53 on (1/2, 1]):

    |t*| <= 1/(2N) + 2^-53.

So max|t| ~= 2^-(K+1); the generator verifies the exact per-cell maximum
(measured 2^-7.0 / 2^-8.0 / 2^-9.0 for N = 64/128/256).

**Rounding of the FMA.** |t| <= 2^-(K+1)(1+eps) implies the single rounding
errs by at most ulp(2^-(K+1))/2 = 2^-(K+55); for N=128 that is 2^-63,
i.e. <= 2^-55 *relative* to a result of magnitude >= 2^-(K+1) — negligible
against the 0.5-ulp final rounding. Crucially the FMA rounding is the *only*
error in t.

**Exact endpoints.** 1/r_0 = 1 and 1/r_N = 1/2 are exact doubles, so:

* j = 0: t = fl(m*1 - 1) = fl(m - 1). For m in [1, 2), m - 1 is exactly
  representable (Sterbenz: 1/2 <= 1 <= m <= 2*1, so the subtraction is
  exact), and an FMA that rounds an exactly-representable value returns it.
  So **t = m - 1 exactly**, and L_0 = (0, 0).
* j = N: t = fl(m/2 - 1). m/2 is exact (power-of-two scale); m in the last
  cell means m/2 in [1 - 1/(4N)·2, 1), and m/2 - 1 is exact by the same
  Sterbenz argument. So **t = m/2 - 1 exactly**, and (after §4's ln2 fold)
  L_N = (0, 0).

These two exact cells are the near-1 treatment (§6).

## 4. sqrt(2) re-centering

If m were used as-is, then x slightly below 1 (m near 2, e = -1) would
compute `(-1)*ln2 + log m` with log m near +ln2 — catastrophic cancellation
against an anchor value of size ~0.69 whose rounding (~2^-54) would dwarf a
result of size ~2^-52. Instead, re-center the mantissa interval to
[sqrt2/2, sqrt2): for anchors r_j >= ~sqrt2, fold a -ln2 into the stored
L_j (at full precision, before the hi/lo split) and have the kernel add 1
to e. Cut index:

    CUT = round(N*(sqrt2 - 1))     (53 for N=128; 27 / 106 for N=64 / 256).

The kernel's branchless form: the compare mask `j >= CUT` is all-ones (-1)
as an integer, and `e -= mask` adds 1 exactly on those lanes.

Consequences, all verified by the generator:

* |L_j| <= max(log r_(CUT-1), ln2 - log r_CUT) ~= ln2/2 = 0.347 for all j.
* The adjusted exponent e' is 0 **iff** x in ~[sqrt2/2, sqrt2), which
  contains every x near 1. Whenever e' != 0,
  |log x| >= ln2 - max|L| - max|t| >= 0.342: the sum e'*ln2 + L + series
  never suffers catastrophic cancellation.
* x just **below** a power of two (m near 2) lands in cell j = N, where the
  folded anchor is L_N = -log(1/2) - ln2 = 0 *exactly* (the generator
  asserts the 300-bit cancellation is identically zero) and t = m/2 - 1 is
  exact: the entire mantissa contribution is the relatively-accurate series.
  x just **above** a power of two lands in cell j = 0 with the same
  property. This is why stress bucket B costs nothing.

## 5. The ln2 pair

Requirement: `A = e * LN2_HI` must be **exact** for every |e| <= 1075 < 2^11.
If LN2_HI carries at most 42 significant bits, the integer product
`e * (LN2_HI * 2^42)` is below 2^11 * 2^42 * ln2 < 2^53, hence exact in
binary64. Since ln2 in [1/2, 1), define

    LN2_HI = round(ln2 * 2^42) / 2^42        = 0x1.62e42fefa3800p-1
    LN2_LO = fl(ln2 - LN2_HI)                = 0x1.ef35793c76730p-45

|LN2_HI - ln2| <= 2^-43, so |LN2_LO| <= 2^-43 and its own rounding is at
most ulp(2^-43)/2 = 2^-96. Generator-verified: |ln2 - (hi+lo)| < 2^-95.
Total representation error in e*ln2 is <= 1075 * 2^-95 < 2^-84, and since
e != 0 implies |result| >= 0.342 (§4), this is < 2^-30 ulp of the result.
`e*LN2_LO` is folded by FMA into the low-order channel: `E = fl(e*LN2_LO +
L_lo)`; |E| <= 1075*2^-43 + 2^-55 < 2^-32, again harmless at the 0.342
result scale (and when e' = 0 it reduces to L_lo <= 2^-55).

## 6. Near x = 1 (and the compensated anchors)

For x = 1 + eps the result ~ eps - eps^2/2 has unbounded relative
sensitivity; any O(2^-54) additive noise from an anchor is catastrophic.
Treatment chosen: **grid alignment on both sides**, from §3 —

* x slightly above 1: e' = 0, j = 0, L = (0,0), t = m - 1 exact;
* x slightly below 1: m near 2, e = -1, j = N >= CUT so e' = 0, L = (0,0),
  t = m/2 - 1 = x - 1 exact (m = 2x here).

In both cases the kernel degenerates to `t + t^2*q(t)` with exact t and no
anchor at all: relative accuracy is limited only by the series (§7) and one
final rounding. The neighbourhood width is the full half-cell, |x-1| <=
1/(2N) = 2^-8 for N=128 — vastly wider than the "few hundred ulps" stress
region. No separate code path, blend, or branch is needed.

Elsewhere the anchor is nonzero and is stored as a hi/lo pair
`L_hi = fl(L_j)`, `L_lo = fl(L_j - L_hi)`; pair error < 2^-104 (generator
assert), so anchors never consume ulp budget. The smallest nonzero anchors
bound the smallest table-path results: min over j of |L_j| / max-cell-|t|
is ~2 (generator: 1.992 / 1.996 / 1.998 for N = 64/128/256), i.e. every
nonzero-anchor cell has |result| >= |L| - |t| >= |t|, at least ~2^-(K+2).
That margin is also the Fast2Sum precondition in §8.

## 7. The series for log(1+t)

From the geometric series 1/(1+u) = sum_{k>=0} (-u)^k (|u| < 1), integrating
term-by-term from 0 to t:

    log(1+t) = t - t^2/2 + t^3/3 - ...  = t + t^2 * q(t),
    q(t) = -1/2 + t/3 - t^2/4 + ... + c_d t^(d-2),   c_k = (-1)^(k+1)/k.

Truncation after total degree d: grouping the tail geometrically,

    |log(1+t) - P_d(t)| <= |t|^(d+1) / ((d+1)(1-|t|)),

i.e. a *relative* error (vs log(1+t) ~ t) of about |t|^d/(d+1). The
generator evaluates the exact maximum relative error of each candidate
polynomial **with coefficients rounded to double** over the full |t| range:

    N=64 : deg 7 -> 2^-52.0   deg 8 -> 2^-59.2
    N=128: deg 6 -> 2^-50.8   deg 7 -> 2^-59.0   deg 8 -> 2^-67.1
    N=256: deg 6 -> 2^-56.8   deg 7 -> 2^-66.0

Because the j=0 / j=N cells rely on pure relative accuracy up to |t| =
2^-(K+1), the series must stay below ~2^-55 relative there; hence the
admissible pairs are (64, 8), (128, 7), (256, 6) — confirmed empirically in
§10, where (128, 6) shows a 4.7-ulp failure exactly in the cell-edge bucket.

**Quasi-minimax variant.** Keeping c1 = 1 and c2 = -1/2 exact (both are
required for relative accuracy at tiny t: an error delta in c2 contributes
only delta*t relative), fit the remainder
`h(t) = (log(1+t) - t + t^2/2)/t^3` by its degree-3 Chebyshev-node
interpolant on [-a, a] (a = padded max|t|, N=128) — near-minimax by standard
approximation theory (`mp.chebyfit`, a generic fitting utility). Result:
total degree 6 with relative error 2^-53.8, a 3-bit (=8x) gain over the
degree-6 Taylor truncation, matching the ~2^(d-2) Chebyshev-vs-Taylor
factor. Measured: max 0.71 ulp (§10) — one FMA cheaper than deg-7 Taylor
but above the deg-7 accuracy; kept as a documented alternative.

Evaluation: Horner in t for q (all FMA), then `p = t^2 * q(t)`. Horner
rounding is a few ulp of |q| ~ 1/2 scaled by t^2, i.e. O(2^-53)*|p| with
|p| <= t^2/2: at most ~2^-70 absolute for N=128 — and it *scales as t^2*,
so it never harms the near-1 relative regime.

## 8. Accumulation order and compensation

Result = A + B + C + p + E with A = e'*LN2_HI (exact, §5), B = L_hi,
C = t, p = series tail, E = fl(e'*LN2_LO + L_lo).

**Fast2Sum lemma** (standard FP arithmetic): if s = fl(a+b) and the exponent
of a is >= that of b (sufficient: |a| >= |b| or a = 0), then the rounding
error a+b-s is exactly representable and equals fl(b - fl(s-a)). Proof
sketch: under the precondition, s-a is exact (Sterbenz-type argument on the
alignment of b against a), and b-(s-a) is the exact low part.

Three variants, all built and measured (§10):

* **ACCUM=0 (plain):** `A + (B + (C + (p + E)))`. Each add can round at the
  scale of the larger operand: the B+... add rounds at ulp(0.35)/2 = 2^-55
  even when the final result is ~0.35-scale; bound ~1.5 ulp; measured max
  0.74 (cell-edge bucket).
* **ACCUM=1 (one Fast2Sum):** s = A + B with err recovered.
  Precondition: A = 0 (then exact trivially: s = B, err = 0) or |A| >=
  LN2_HI = 0.693 > 0.347 >= |B|. Then `s + (C + (p + (E + err)))`.
  Remaining unprotected rounding: the C-add, <= ulp(2^-(K+1))/2, which is
  <= 0.5 ulp of the smallest table-path result 2^-(K+2) (§6 margin); bound
  ~1.05 ulp worst case; measured max 0.52.
* **ACCUM=2 (two Fast2Sums):** additionally s2 = s + t with err2 recovered.
  Precondition |s| >= |t| holds because either e' != 0 (|s| >= 0.34) or
  s = B = L_j exactly (err = 0) with the generator-verified margin
  |L_j| >= ~2*max-cell-|t| (§6); in the exact-anchor cells s = 0 and
  Fast2Sum(0, t) is trivially exact. Then
  `r = s2 + (p + (E + (err + err2)))`. Every addition before the final one
  now either is error-free or operates at the tail scale (<= 2^-17), whose
  roundings (<= 2^-70) and the series/pair errors sum to < 0.1 ulp of even
  the smallest results: provable bound ~0.6 ulp; measured max 0.52.

Sign detail for log(1) = +0: x = 1 gives e' = 0, all of A, B, err, err2,
E, t, p equal +0.0, and (+0)+(+0) = +0 in round-to-nearest, so the kernel
returns +0 bitwise.

## 9. Special values and subnormals

Fast-path predicate (one unsigned compare): x is a finite positive normal
iff `bits(x) - 0x0010000000000000 <= 0x7FDFFFFFFFFFFFFF` (this window is
exactly [0x0010.., 0x7FEF..F]; everything else — +/-0, subnormals, all
negatives, +/-inf, NaN — wraps above it). If any lane fails, a noinline
slow path:

* positive subnormal (0 < bits < 0x0010..): lane-scaled by 2^64 (exact, §1),
  e offset -64 via a masked integer add;
* +/-0 -> -inf; negative (sign set, not -0, not NaN) -> quiet NaN;
  +inf -> +inf; NaN -> x+x (propagates, quiets signaling NaNs); patches
  applied by `vbslq_f64` in an order that ends with NaN (so a NaN is never
  misclassified as negative — NaN lanes are explicitly excluded from the
  negative mask anyway).

Unusable lanes are parked at 1.0 before the core call so no spurious traps
or garbage indexing can occur.

## 10. Design-space exploration (measured, Apple M1, clang -O3)

Max ULP vs the correctly-rounded 300-bit reference, per bucket
(A near-1, B near-powers-of-2, C subnormal, D log-uniform, E cell-edge
stress); ns/element, best of repeated runs (scalar `std::log` baseline
~2.72-2.85 ns/elem on the same data):

| N   | deg | poly    | ACCUM | UNROLL | A     | B     | C     | D     | E     | hot   | stream |
|-----|-----|---------|-------|--------|-------|-------|-------|-------|-------|-------|--------|
| 64  | 8   | Taylor  | 2     | 2      | .5000 | .4931 | .4991 | .4997 | .5428 | 1.78  | 1.79   |
| 128 | 6   | Taylor  | 2     | 2      | .5000 | .4931 | .4991 | .4997 | **4.71** | 1.88 | 2.15 |
| 128 | 7   | Taylor  | 2     | 2      | .5000 | .4931 | .4991 | .4997 | .5190 | 1.62  | 1.63   |
| 128 | 6   | minimax | 2     | 2      | .5913 | .4931 | .4991 | .4997 | .7108 | 1.72  | 1.97   |
| 256 | 6   | Taylor  | 2     | 2      | .5000 | .4931 | .4991 | .4997 | .5287 | 2.26  | 2.20   |
| 256 | 7   | Taylor  | 2     | 2      | .5000 | .4931 | .4991 | .4997 | .4998 | 1.73  | 1.78   |
| 128 | 7   | Taylor  | 0     | 2      | .5000 | .4931 | .4991 | .5145 | .7445 | 1.37  | 1.37   |
| 128 | 7   | Taylor  | 1     | 2      | .5000 | .4931 | .4991 | .4997 | .5190 | 1.51  | 1.50   |
| 128 | 7   | Taylor  | 2     | 1      | .5000 | .4931 | .4991 | .4997 | .5190 | 1.67  | 1.70   |

Observations:

* (128, 6, Taylor) fails exactly where §7 predicts (2^-50.8 relative ->
  ~4.7 ulp at the j=0/j=N cell edges); everything with series error below
  ~2^-55 sits at ~0.5 ulp, i.e. within one rounding of correctly-rounded.
* The compensation ladder behaves as derived: plain 0.74 max, one/two
  Fast2Sums 0.52; the second Fast2Sum is not visible in these buckets but
  tightens the provable bound (~1.05 -> ~0.6 ulp) for 0.11 ns.
* N=256 halves |t| but the larger table costs more than the saved FMA
  (gather-dominated); N=64 needs deg 8 and is slower, not faster.
* Unroll x2 buys ~0.05 ns (more ILP for the two software-gather chains).

**Operating point (defaults):** `N=128, Taylor deg 7, ACCUM=2, UNROLL=2` —
max 0.519 ulp over every bucket, provably < 1 ulp, 1.62/1.63 ns per element
hot/stream vs ~2.75-2.85 scalar (~1.7x). `ACCUM=1` is the documented speed
knob (identical measured accuracy, ~1.50 ns, weaker worst-case proof);
table row retained so the trade-off stays visible.

## 11. Table layout and gather

Rows `{L_hi, L_lo, R, pad}` (32 B, 64-B aligned): a row never straddles a
cache line, the (L_hi, L_lo) pair is one `vld1q_f64`, R one 8-B load, and
lane merging is `vuzp1q/vuzp2q_f64` + `vcombine_f64` — NEON has no gather,
so this is the two-loads-per-lane composition of SPEC §4. Total 129 rows =
4128 B, resident in L1 alongside the working set.

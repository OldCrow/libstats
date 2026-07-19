# Divergence Audit — clean-room NEON `sin`/`cos` (attempt_A) vs ARM optimized-routines

Date: 2026-07-19
Author: orchestrator (not the clean-room implementer). Purpose: document that the
clean-room implementation in `attempt_A/` does not reuse the expression of any existing
vector trigonometric implementation, and record exactly where the two coincide and why.
Note the motivation differs from the erf audit: the comparison target here is
MIT-licensed (ARM optimized-routines, upstream of the family libstats' other kernels
derive from), so this audit protects *independence of derivation* (the basis for the
accuracy claims), not license compliance.

## Method
The clean-room implementer authored `attempt_A/` in isolation from a functional spec
(SPEC.md) only. This audit was performed *after* authorship by the orchestrator, who
separately fetched ARM optimized-routines `math/aarch64/advsimd/sin.c` and `cos.c`
(master, retrieved 2026-07-19) for comparison. The implementer never saw those sources;
this comparison does not feed back into the authored code.

### Provenance incidents (disclosed)
1. **Workspace collision.** A second, independent attempt at the same task briefly ran
   in the same directory (orchestrator error: the first agent appeared
   permission-stalled and was re-spawned). While diagnosing the collision, the
   attempt_A author saw a few headline choices of attempt B (its D_max = 2^20,
   32-bit split parts, chebyfit usage). Every corresponding attempt_A choice was on
   disk with earlier mtimes (01:44–01:45 vs 01:55+) before the exposure; details in
   attempt_A/DERIVATION.md's provenance note. Attempt B was stopped and its artifacts
   preserved unused in `attempt_B_stopped/`; attempt B's own independence is
   *not* certified (it began execution amid attempt A's existing files) and it must
   not be integrated without its own audit.
2. Neither attempt saw ARM's sources, libstats' kernels, or any other implementation.

## Two categories of coincidence, and why they are not copying
1. **Mathematically forced.**
   - `2/π` head constant: attempt_A `TWO_OVER_PI_D = 0x1.45f306dc9c883p-1`; ARM
     `inv_pi = 0x1.45f306dc9c883p-2`. Same mantissa, different exponent — forced:
     both are the correctly-rounded doubles of 1/π scaled by an exact power of two;
     rounding commutes with binary scaling, so any correct derivation produces this
     mantissa.
   - Leading digits of polynomial coefficients converge to the Taylor values
     (−1/6, 1/120, …; −1/2, 1/24, …) in both, as any correct fit must. The tail
     digits — the expressive content of a minimax fit — differ in **every**
     coefficient (see below).
   - `D_max = 2^23` in both. Convergent, not copied: ARM's `range_val = 0x1p23` bounds
     its scheme's fast path; attempt_A *derives* 2^23 from its own 30-significant-bit
     split parts (products `n·p_k` exact for `|n| < 2^53−30 = 2^23`, and
     `n_max = round(2^23·2/π) = 5,340,354 < 2^23`), documented in DERIVATION.md §2
     before any exposure to anything. 2^23 is a natural binade bound for
     double-precision trig reduction; the rationales are different and independent.
   - Applying quadrant sign by shifting a quadrant bit to position 63 and XOR-ing:
     present in both, but it is the generic IEEE sign-bit idiom and was explicitly
     suggested in SPEC.md §4 ("flip signs via XOR of the IEEE sign bit") as an
     available generic technique.

2. **Idiomatic / spec-given technique.** FMA-chained multi-constant subtraction for
   range reduction, Horner-in-r² polynomial evaluation, round-then-convert quadrant
   extraction — standard vector-math technique, described generically in the spec.
   Methods are not the protected content; their expression is, and the expressions
   differ throughout (below).

## Material divergences (free choices — all differ)
- **Reduction period**: ARM reduces modulo **π** (`n = rint(x/π)`, quadrant = n mod 2,
  one sin core; cos via a half-period offset added before rounding). attempt_A reduces
  modulo **π/2** (`n = round(x·2/π)`, quadrant = n mod 4, **two** cores with
  branch-free core-swap). Structurally different schemes with different quadrant
  logic, different fit intervals, different everything downstream.
- **Split-constant construction**: ARM uses a 3-part extended-precision π
  (`pi_1 = 0x1.921fb54442d18p+1` — a full 53-bit head — plus 2 tails); products
  `n·pi_1` are *not* exact and the scheme absorbs that in the FMA chain. attempt_A
  uses a 4-part **exact-product** split of π/2 with 30-significant-bit parts chosen
  to have ≥23 trailing zero mantissa bits (`0x1.921fb54800000p+0,
  -0x1.de973dc800000p-31, -0x1.9d9cceb800000p-62, -0x1.1fc8f8cbb5bf7p-93`), a
  Sterbenz-exactness argument per step, and an explicitly **compensated residual
  (r, rlo)** recovered via a second FMS per step. No constant matches; the method,
  the bit-width choice, and the error analysis are attempt_A's own.
- **Rounding mode for n**: ARM `vrndaq` (round-away); attempt_A `vrndnq`
  (round-to-nearest-even, with a derived argument for why plain rounding suffices,
  DERIVATION.md §3.4).
- **Polynomial coefficients**: both use degree-6 odd sin cores, but on different
  intervals ([0,(π/4)²] vs ARM's π-reduction interval), and every coefficient's tail
  digits differ (e.g. attempt_A `-0x1.a01a01a019938p-13` vs ARM
  `-0x1.a01a019936f27p-13`). attempt_A additionally has an independent degree-6
  **cos core** (`1 + u·Q̃(u)` with an exact `1 − u/2` head/tail split and an
  `−r·rlo` first-order compensation term) that has no counterpart in ARM's scheme
  at all.
- **Special handling**: attempt_A's sin-only tiny-|x| mask at 2^−26 preserving −0;
  scalar-oracle lane patching for out-of-domain/inf/NaN; fused sincos sharing both
  cores. ARM instead branches to a table-based large-range reduction path.
- **Accuracy class**: ARM documents ~2.7–2.8 (+0.5) ULP bounds; attempt_A measures
  0.78 ULP uniform / 0.50 ULP stress maxima. The clean-room implementation is not a
  re-expression of ARM's design point — it occupies a different (more accurate,
  compensated) point in the design space, which is the program's purpose.

## Scope and verdict
Compared against: ARM optimized-routines advsimd `sin.c`/`cos.c` (the upstream family
of libstats' existing SIMD kernels) and libstats' current `vector_cos_neon` (naive
Taylor + single-constant 2π reduction — shares nothing with attempt_A beyond Horner
FMA evaluation). SLEEF and other libraries were not individually compared; given the
structural divergence (π/2 quadrant scheme, exact-product compensated reduction, dual
cores) and fully distinct constants, the conclusion is robust: **attempt_A contains no
copied expression; all coincidences are mathematically forced values or spec-provided
generic techniques; every free design choice differs.** Attempt B remains uncertified
and unused.

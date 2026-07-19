# Divergence Audit — clean-room NEON `log` vs ARM optimized-routines `advsimd/log.c`

Date: 2026-07-19
Author: orchestrator (not the clean-room implementer). Purpose: document that the
clean-room implementation in this directory does not reuse the expression of any
existing vector logarithm, and record exactly where the two coincide and why. As with
the trig audit, the comparison target is MIT-licensed; this audit protects
*independence of derivation* (the basis of the accuracy claims), not license
compliance.

## Method
The clean-room implementer authored everything here in isolation from SPEC.md only,
and reported no consultation of any implementation. This audit was performed *after*
authorship by the orchestrator, who separately fetched ARM optimized-routines
`math/aarch64/advsimd/log.c` (master, retrieved 2026-07-19). The implementer never saw
it; this comparison does not feed back into the authored code. The orchestrator also
compared against libstats' current `vector_log_neon` (SLEEF-family `(m−1)/(m+1)`
atanh-series kernel), which the implementer likewise never saw.

## Important disclosure: the shared method was spec-prescribed
Both this implementation and ARM's `log.c` belong to the same well-known method
family: an anchored table storing a reciprocal and a log value per cell, residual
`t = m·(1/r) − 1` formed by multiplication (no division), plus a short series in `t`.
SPEC.md §3 prescribed that family (written by the orchestrator as a generic numerical
technique, stated without reference to any implementation). Methods are not the
protected content; what this audit certifies is that the *expression* — constants,
table contents and layout, centering mechanism, accumulation scheme, special-case
structure — is independently derived. The implementer's report notes they deliberately
derived the anchor identity (`L = −log(fl(1/r))` making the decomposition exact)
rather than recalling "how it is usually arranged"; that identity and its exactness
proof are in DERIVATION.md.

## Coincidences, and why they are not copying
1. **Mathematically forced.**
   - Series coefficients: the clean-room kernel uses truncated Taylor for
     `log(1+t)` — `(−1)^{k+1}/k` — which are universal constants any correct
     derivation produces. ARM instead uses a degree-4 *minimax* set
     (`-0x1.ffffffffffff7p-2, 0x1.55555555170d4p-2, …`) whose digits appear nowhere
     in the clean-room code. No coefficient value is shared beyond the forced
     leading digits any log kernel exhibits.
   - `ln2`: both need it; ARM stores the single correctly-rounded double
     `0x1.62e42fefa39efp-1`. The clean-room kernel stores a **42-bit-head + tail
     split** (chosen so `e·ln2_hi` is exact for all |e| ≤ 1075) — a different
     representation with a documented exactness proof; the correctly-rounded value
     itself is a universal constant.
   - **Table size N = 128 in both** (ARM's `V_LOG_TABLE_BITS` = 7 upstream).
     Convergent optimum, not copied: the clean-room implementer measured N = 64
     (needs degree 8), 128, and 256 (gather-dominated, slower) and chose 128 from
     that data; the design-space table is in DERIVATION.md §10. With a 52-bit
     mantissa and a gather-cost cliff on this core, the practical optimum is narrow;
     coincidence here is expected.
2. **Idiomatic / spec-given technique.** Reciprocal-table residual without division,
   top-mantissa-bit indexing, software gather via paired loads + `vuzp`, Horner FMA
   evaluation — all named in SPEC.md §3–4 as generic techniques, all standard.

## Material divergences (free choices — all differ)
- **Centering mechanism**: ARM applies a magic subtractive offset
  (`0x3fe6900900000000`) to the raw bit pattern before extracting index and exponent.
  The clean-room kernel has **no offset constant at all**: √2-centering is *folded
  into the table* (anchors above ~√2 store `L − ln2` and a branchless mask bumps `e`),
  with both interval ends grid-aligned (`r_0 = 1`, `r_N = 2`, exact reciprocals,
  `L = (0,0)` exactly). Different mechanism, no shared constant, and the alignment
  is what makes the near-1 neighbourhood degenerate to a pure relatively-accurate
  series — a feature ARM's scheme does not have.
- **Table entry contents/layout**: ARM `{invc, logc}` (16 B, single-double log).
  Clean-room `{L_hi, L_lo, R, pad}` (32 B, 64-B aligned, **compensated** log anchor).
  The compensation is the central accuracy choice (same philosophy as this program's
  erf kernel) and has no counterpart in ARM's entries.
- **Accumulation scheme**: ARM sums with plain FMA chains and a single ln2 multiply.
  Clean-room uses two proved-exact Fast2Sum steps with a generator-verified table
  margin precondition, tails summed small-to-large — the DERIVATION.md §-level error
  analysis is the implementer's own.
- **Series**: degree 7 Taylor in `t + t²·q(t)` form (keeping c1, c2 exact for the
  near-1 regime) vs ARM's degree-4 minimax in odd/even split form. Different degree,
  different form, different digits, different rationale.
- **Index extraction**: clean-room `(frac + 2^44) >> 45` round-to-nearest of
  `N(m−1)` with an exactness proof; ARM shift/mask of offset bits (truncation).
  Different arithmetic, different behavior at cell edges (the clean-room stress
  bucket E exists precisely to test this).
- **Subnormal handling**: clean-room exact `2^64` prescale with `e − 64` adjustment,
  vectorized; ARM routes specials to its scalar special-case path.
- **Accuracy class**: ARM documents 1.67 + 0.5 ULP; the clean-room kernel measures
  **0.52 ULP max** across all buckets (verified by the orchestrator on a clean
  rebuild). Different operating point — the compensated-anchor design is not a
  re-expression of ARM's.
- **vs libstats' current kernel** (SLEEF family): nothing shared — that kernel is
  division-based `(m−1)/(m+1)` with an atanh series and no table; the clean-room
  kernel is division-free and table-anchored, and beats it on both axes
  (0.52 vs 2.00 ULP max; 1.66 vs 2.88 ns/element).

## Scope and verdict
Compared against ARM optimized-routines advsimd `log.c` (closest structural relative,
same method family) and libstats' current `vector_log_neon`. SLEEF and other
libraries were not individually compared; given that every expressive choice examined
differs and the only shared values are mathematically forced or convergent-by-
measurement, the conclusion is robust: **no copied expression; the shared method
family was prescribed by the spec as a generic technique; every free design choice
differs, and several (compensated anchors, exact end-anchored centering, proved
Fast2Sum pipeline) have no counterpart in the comparison target.**

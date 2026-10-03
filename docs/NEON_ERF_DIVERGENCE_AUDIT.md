# Divergence Audit — clean-room NEON `erf` vs glibc `erf_advsimd`

Date: 2026-07-19
Author: orchestrator (not the clean-room implementer). Purpose: document, for
Issue #67, that the clean-room implementation in this directory does not reuse the
copyrightable *expression* of glibc's LGPL-2.1+ `sysdeps/aarch64/fpu/erf_advsimd.c`,
and to record exactly where the two coincide and why.

## Method
The clean-room child (`erf-cleanroom`) authored the implementation in isolation, from a
functional spec only, with no access to any existing `erf` implementation (see SPEC.md
§2 and the child's honesty note). This audit was performed *after* authorship by the
orchestrator, who had separately read glibc's `erf_advsimd.c`. The child never saw that
source; this comparison does not feed back into the authored code.

## Two categories of coincidence, and why they are not copying
1. **Mathematically forced** — the numeric Taylor-coefficient *values*. `erf` has one
   Taylor expansion; any correct derivation yields the same rational coefficients
   (1/3, 2/15, 2/5, 2/9, 2/45, …). These are facts, not authorship — not copyrightable,
   and coincidence is unavoidable. The child derived them independently from the Hermite
   recurrence and verified them symbolically (`sympy.series`) and numerically (mpmath).
2. **Idiomatic / spec-given technique** — a table of `{erf(r), scale=erf'(r)}` plus a
   local Taylor correction, fetched by the NEON software-gather `2×vld1q + vuzp1/vuzp2`.
   The *method* was described in the spec (a generic numerical technique, stated without
   reference to any implementation); methods/algorithms are not copyrightable, only their
   expression is. The `vuzp` software-gather is the standard NEON idiom and is already
   independently present in this project's MIT-licensed exp/log kernels.

Everything that is a *free, expressive* choice differs. Details below.

## Comparison of free design choices

- Grid spacing: glibc 1/128; clean-room **1/256** (h = 2⁻⁸). Differ.
- Table size: glibc 769 entries; clean-room **1537**. Differ.
- Table entry layout: glibc `{erf(r), scale}` (16 B); clean-room
  **`{E_hi, S, E_lo, r}` (32 B)** — a compensated (double-double) `erf(r)` plus a stored
  grid point. Differ (and the compensation is why clean-room hits 1 ULP where glibc is
  2.29 ULP).
- Saturation bound: glibc 5.9921875 (= 6 − 1/128, a grid-aligned value); clean-room
  **5.921587195794507**, the exact smallest double whose `erf` rounds to 1, found by
  bit-pattern bisection. Different value, different rationale.
- Index extraction: glibc uses the "add a large power of two, read low mantissa bits"
  shift trick; clean-room uses **`fmul 2⁸` + `fcvtns`** (round-to-nearest convert) and
  explicitly measured and *rejected* the magic-add approach. Differ.
- Series length: glibc 5 terms; clean-room **5 terms** — but independently arrived at,
  after exploring N = 4..8 and choosing 5 as the shortest meeting < 1 ULP, with its own
  documented reason (odd N is efficient because even-index coefficients vanish at r = 0).
  This is *convergence on a math-optimal parameter*, not copying; the value 5 is close to
  forced once the accuracy target and grid are fixed.
- Polynomial organization: glibc `erf(r) + scale·(d − d²·y)` with `y = p1 + d·p2 + …` and
  inline per-`r` FMA constants; clean-room `E + S·(d + Σ c_k(r) d^k)` with separate
  `coef_cN(r)` helpers evaluated by Horner in `w = r²`. Different factoring, different
  sign convention, different code shape.
- Naming / structure: glibc `struct data`, `_ZGVnN2v_erf`, `lookup()`; clean-room
  `erf_f64x2`, `coef_cN`, `poly_t<N>`, templated `Cfg`/`N`/`COMP` design-space harness.
  No shared identifiers.
- Accuracy outcome: glibc 2.29 ULP; clean-room **1.00 ULP** (measured, 22008-point
  correctly-rounded reference; independently reproduced by the orchestrator).

## Coincidences, itemized
- The Taylor-coefficient rationals (1/3, 2/15, 2/5, 2/9, 2/45, …): forced by the
  mathematics — appear in any correct erf table+Taylor derivation.
- The general table+Taylor method and the `vuzp` software-gather idiom: spec-given
  generic technique / standard NEON idiom (also used independently by this project's
  MIT-derived exp/log kernels).
- Series length 5: independently chosen from measured data; near-forced by the
  accuracy/grid targets.
No verbatim code, no shared identifiers, and no shared *expressive* constant (grid size,
table layout, saturation value, index constant) were found.

## Conclusion
The clean-room implementation is consistent with independent creation. Its only overlaps
with glibc's `erf_advsimd.c` are (a) mathematically forced coefficient values, and
(b) a generic, spec-described method plus a standard NEON idiom — none of which is
protectable expression. Every free design decision differs, and the result is measurably
more accurate. On the engineering/process evidence, replacing the tainted
`vector_erf_neon` with this implementation resolves the provenance concern in Issue #67.
(This is an engineering and process record, not a legal opinion.)

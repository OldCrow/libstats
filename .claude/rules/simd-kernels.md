---
paths:
  - "src/simd_*.cpp"
  - "include/libstats/platform/simd*.h"
  - "include/libstats/common/simd_*.h"
---

## SIMD kernel conventions

- **A SIMD kernel must never re-read its input array after the corresponding
  store.** Decide every edge fixup from already-loaded registers. This binds
  the `VectorOps` kernel layer, where in-place calls are legal
  (`LogSpaceOps::logSumExpArrayFallback` calls `vector_exp` with
  `a == result`); the distribution batch span overloads are the opposite case
  and promise no in-place safety at all (#112). In-place legality is a
  `VectorOps` property; it stops at that layer.
- **Accuracy claims hold only for tiers validated on native silicon.**
  `LIBSTATS_MAX_SIMD_TIER` (cmake/SIMDDetection.cmake) caps the highest
  compiled x86 tier so lower tiers can run natively on capable hardware; the
  first-ever native SSE2 run is what exposed #74, invisible under Rosetta for
  years.
- **Gather-vs-polynomial transcendentals are settled** (#33,
  `docs/SIMD_BENCHMARK_RESULTS.md`): x86 hardware gather is too expensive to
  beat a polynomial; NEON is the opposite, where a table lookup is nearly
  free. Table kernels are a NEON technique here, not an x86 one.

## FP-contraction rule

**A new error-free transform (Kahan/Neumaier, TwoSum/Fast2Sum, `fma(a,b,-a*b)`
residuals) must spell every intended fusion explicitly, or scope
`-ffp-contract=off` to that file.** No `-ffp-contract` flag is set anywhere in
this build, so every TU takes its compiler default (GCC `fast`, AppleClang
`on`, MSVC/clang-cl off); an unspelled contraction silently breaks the exactness
the transform's proof assumes. The three compensated sequences in
`src/simd_neon.cpp` are safe today (#84, audited 2026-08-16) because every
fusion is an explicit `vfmaq_f64`/`vfmsq_f64` and the one remaining
multiply-then-add is exact by construction.

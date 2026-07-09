# libstats Validation History

Historical per-version validation matrices and SIMD speedup benchmarks, extracted from
`AGENTS.md` to keep that file lean for session start. `AGENTS.md` keeps only the
current-release validation matrix and current deferred items; this file holds everything
prior. For detailed per-version change narratives, see `CHANGELOG.md` (auto-generated via
git-cliff). For SIMD methodology, see `docs/SIMD_OPTIMIZATION_REFERENCE.md` and
`docs/SIMD_BENCHMARK_RESULTS.md`.

## Validation matrices by release

### v2.0.0 — validation target (three machines)

Ivy Bridge / macOS Catalina dropped from the ecosystem in v2.0.0 (Catalina EOL;
minimum macOS raised to 13 Ventura).

| Machine | SIMD | Target | Notes |
|---|---|---|---|
| Kaby Lake (2017 MBP) | AVX2+FMA | 46/46 ✅ | Audit remediation complete (2026-07-01) |
| Mac Mini M1 | NEON | 46/46 ✅ | Audit remediation complete (2026-07-01) |
| Asus TUF A16 (Windows) | AVX-512 | 46/46 ✅ | Re-validation required after audit remediation |

### v1.5.2 — final v1.x release (four machines)

| Machine | SIMD | Correctness | Notes |
|---|---|---|---|
| Kaby Lake (2017 MBP) | AVX2+FMA | 39/39 ✅ | |
| Ivy Bridge (2012 MBP) | AVX | 38/38 ✅ | (last version with Catalina) |
| Mac Mini M1 | NEON | 39/39 ✅ | |
| Asus TUF A16 (Windows) | AVX-512 | 39/39 ✅ | |

### v1.5.1 — validated on all four machines

`simd_verification` reports **geometric mean speedups** per operation type (PDF/LogPDF/CDF)
and per primitive vector op, not a single composite. See `tools/simd_verification.cpp` for rationale.

| Machine | SIMD | Correctness | Total suite | simd_verification | PDF geomean | LogPDF geomean | CDF geomean |
|---|---|---|---|---|---|---|---|
| Kaby Lake (2017 MBP) | AVX2+FMA | 39/39 ✅ | 61 | 61/61 ✅ | 8.0x | 9.6x | 3.3x |
| Mac Mini M1 | NEON | 39/39 ✅ | 61 | 61/61 ✅ | 5.9x | 7.3x | 3.1x |
| Asus TUF A16 (Windows) | AVX-512 | 39/39 ✅ | 61 | 61/61 ✅ | 4.8x | 5.1x | 2.2x |

Kaby Lake primitive vector op speedups (v1.5.0 Phase 1+2): VectorExp 3.4x, VectorLog 1.7x, VectorErf 2.5x, VectorCos 4.9x.
Mac Mini M1 primitive vector op speedups (v1.5.0 Phase 3): VectorExp 2.1x, VectorLog 1.8x, VectorErf 8.0x, VectorCos 3.0x.
Asus TUF A16 primitive vector op speedups (v1.5.0 Phase 4): VectorExp 5.0x, VectorLog 3.9x, VectorErf 1.3x, VectorCos 8.5x.

### v1.4.0 baseline — all four machines

| Machine | SIMD | Correctness | Total suite | simd_verification | Overall |
|---|---|---|---|---|---|
| Kaby Lake (2017 MBP) | AVX2 | 39/39 ✅ | 59 | 54/54 ✅ | 3.35x |
| Mac Mini M1 | NEON | 39/39 ✅ | 59 | 54/54 ✅ | 2.31x |
| Asus TUF A16 (Windows) | AVX-512 | 39/39 ✅ | 59 | 54/54 ✅ | 1.64x |

**Total suite counts differ by machine (v1.5.0):**
- Kaby Lake (61): v1.5.0 adds VonMises distribution rows + 4 primitive vector op rows to `simd_verification`.
- Mac Mini M1 (61): Phase 3 validated ✅.
- Asus TUF A16 (61): Phase 4 validated ✅.

> **v2.0.0:** macOS minimum raised to 13 Ventura. Ivy Bridge / Catalina support dropped.
> `CROSS_PLATFORM` build type and `LIBSTATS_HAS_REQUIRES_EXPRESSIONS` removed.
> Alternate LLVM compiler infrastructure removed; use system AppleClang.

## SIMD batch operation speedups (historical)

### Ivy Bridge AVX — v1.5.0, historical
v1.5.0 results on Ivy Bridge AVX (61/61 simd_verification ✅): PDF geomean 5.6x, LogPDF 6.0x, CDF 2.6x.
Primitive ops: VectorExp 2.2x, VectorLog 1.3x, VectorErf 1.7x, VectorCos 11.0x.

Selected per-distribution speedups:

| Distribution | Op | Speedup |
|---|---|---|
| Uniform | PDF | 122.4x |
| Uniform | LogPDF | 118.4x |
| Uniform | CDF | 27.0x |
| Gaussian | LogPDF | 44.1x |
| Exponential | LogPDF | 35.4x |
| VonMises | LogPDF | 18.2x |
| Exponential | PDF | 14.5x |
| Exponential | CDF | 7.5x |
| Gamma | PDF | 8.2x |

### v1.0.0 validation (historical)

Ivy Bridge AVX (historical — Catalina support dropped in v2.0.0):
- correctness suite: 34/34 PASS
- `simd_verification`: 54/54 PASS, overall 4.10x
- new-distribution speedups: Chi-squared PDF 9.5x/LogPDF 7.0x, Student's t PDF 7.3x/LogPDF 7.6x,
  Beta PDF 4.6x/LogPDF 4.4x

Asus TUF A16 (Windows, AVX-512 — first AVX-512 validation):
- correctness suite: 33/33 PASS (GTest available via vcpkg gtest:x64-windows 1.17.0)
- `simd_verification`: 54/54 PASS, overall 1.64x
- AVX-512 arithmetic/log-space paths: Gaussian LogPDF 21.9x, Exponential LogPDF 11.8x,
  Uniform LogPDF 7.5x — strong where transcendentals are not involved
- Overall speedup limited by transcendental delegation to AVX (see Deferred Items in AGENTS.md)

Kaby Lake AVX2 (2017 MBP):
- correctness suite: 33/33 PASS
- `simd_verification`: 54/54 PASS, overall 3.49x
- new-distribution speedups: Chi-squared PDF 13.8x/LogPDF 10.5x, Student's t PDF 6.3x/LogPDF 18.4x,
  Beta PDF 5.3x/LogPDF 4.1x

All four machines validated at v1.0.0 (Ivy Bridge/Catalina dropped in v2.0.0).

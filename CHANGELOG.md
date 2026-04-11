# Changelog

## [1.0.0] - 2026-04-11

### Added
- `ChiSquaredDistribution`, `StudentTDistribution`, and `BetaDistribution`
- `detail::digamma(x)` and `detail::inverse_beta_i(p, a, b)` in `math_utils`
- Direct tests and SIMD verification coverage for all three new distributions
- `CHI_SQUARED` added to `DistributionType` enum, `distribution_characteristics`,
  `dispatch_utils`, and all tool distribution lists

### Improved
- Phase 6A SIMD batch paths for Exponential, Gamma, and Uniform
- Dispatch metadata (`getDistributionSpecificParallelThreshold`) now covers all
  9 distributions with explicit cases; `distributionTypeToString` likewise complete
- Documentation and example surface aligned with the current 9-distribution library
- `distribution_families_demo.cpp` added; per-distribution benchmark files removed
  from the examples build
- Explanatory comments added to Gamma, Beta, and Student's t CDF batch
  implementations documenting why scalar special functions cannot be vectorized

### Validation (all four machines, 54/54 SIMD tests)

| Machine | SIMD | Correctness | simd_verification | Speedup |
|---|---|---|---|
| Ivy Bridge (2012 MBP) | AVX | 34/34 ✅ | 54/54 ✅ | 4.10x |
| Kaby Lake (2017 MBP) | AVX2 | 33/33 ✅ | 54/54 ✅ | 3.49x |
| Mac Mini M1 | NEON | 33/33 ✅ | 54/54 ✅ | 2.31x |
| Asus TUF A16 (Windows) | AVX-512/MSVC | 33/33 ✅ | 54/54 ✅ | 1.64x |

---

## Earlier milestones

### Phase 6A — SIMD batch ops for non-Gaussian distributions
Added vectorized `BatchUnsafeImpl` kernels to Exponential (PDF/LogPDF/CDF), Gamma
(PDF/LogPDF), and Uniform (CDF), using the compute+fixup pattern established in
`src/gaussian.cpp`. Speedups on Ivy Bridge AVX: Exponential LogPDF 20.8x,
Exponential PDF/CDF ~10x, Gamma PDF 9.7x, Uniform CDF 25.2x.
Overall `simd_verification` speedup improved from 3.84x to 4.10x.

### Phase 5 — Header optimization and namespace consolidation
Primary namespace changed from `libstats` to `stats`; backward-compatibility alias
`namespace libstats = stats` retained. Header dependency graph cleaned up with
forward-declaration headers and consolidated includes. Compilation overhead reduced.

### Phase 4 — Complete 6-distribution library, SIMD verification
All six core distributions fully implemented with PDF/CDF/quantile/sampling/MLE:
Gaussian, Exponential, Uniform, Poisson, Discrete, Gamma. `simd_verification` tool
validates both correctness and measured speedups. Cross-machine validation completed
on Ivy Bridge (AVX), Kaby Lake (AVX2), M1 (NEON), and Linux CI (AVX2).

### Phase 3 — Performance dispatch infrastructure
`PerformanceDispatcher`, `PerformanceHistory`, and `SystemCapabilities` added.
Architecture-aware parallel thresholds, work-stealing pool, and adaptive strategy
selection via learned performance history.

### Phase 1–2 — Foundation and core infrastructure
Initial library structure: SIMD detection and dispatch (SSE2/AVX/AVX2/NEON),
thread pool, safety utilities, numerical constants, distribution base class, and
fully working `GaussianDistribution` as the reference implementation.

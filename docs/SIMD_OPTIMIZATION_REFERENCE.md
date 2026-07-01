# SIMD Optimization Reference

This document records SIMD design decisions for libstats v2.x.

## Current status

Resolved work from v1.4.0 through v2.0.0:

- `vector_cos` across SSE2, AVX, AVX2, NEON, and AVX-512
- AVX2+FMA native `vector_exp`, `vector_log`, and `vector_cos`
- high-accuracy x86 `vector_erf` based on musl-style rational polynomial regions
- NEON native `vector_exp`, `vector_log`, and table-based `vector_erf`
- AVX-512 native `vector_exp`, `vector_log`, `vector_erf`, and `vector_cos`
- dispatch thresholds calibrated per architecture and operation

## Active SIMD tiers

| Tier | Typical machines | Notes |
|---|---|---|
| SSE2/AVX/AVX2+FMA | Intel Haswell/Kaby Lake and newer | primary x86 macOS baseline |
| NEON | Apple Silicon | M1 validation path |
| AVX-512 | AMD Zen 4 / supported Intel | Windows validation path |

## Deferred work

### vector_floor and vector_blend

Deferred. These primitives would enable branchless Discrete CDF and some Uniform paths, but existing batch-path amortisation already gives large speedups. The expected benefit does not justify a new cross-backend primitive pass before v2.x releases.

### vector_lgamma

Deferred. A correct vectorised log-gamma is complex and has limited immediate distribution impact compared with exp/log/erf/cos.

### SVE

Deferred. No validation hardware in the project ecosystem.

### SSE4.1 tier

Deferred. The v2.x macOS x86 baseline effectively has SSE4.1, but Linux x86 CI still benefits from a simple SSE2 fallback. A dedicated SSE4.1 tier adds maintenance cost for small benefit.

## Validation tools

```bash
./build/tools/system_inspector --quick
./build/tools/simd_verification
```

`simd_verification` reports correctness and per-operation geometric mean speedups. Do not compare raw timing numbers across architectures.

## References

- SLEEF: vector exp/log/cos approximation inspiration
- musl libc: high-accuracy erf rational approximation inspiration
- Agner Fog optimisation manuals: SIMD and instruction-level performance guidance

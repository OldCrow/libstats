# SIMD Benchmark Results

This document records SIMD benchmark results for libstats. Treat older sections as historical baselines, not current release targets.

## v2.x validation targets

Current release validation targets:

| Machine class | SIMD | Validation goal |
|---|---|---|
| Intel Kaby Lake | AVX2+FMA | correctness + `simd_verification` |
| Apple Silicon M1 | NEON | correctness + `simd_verification` |
| AMD Zen 4 | AVX-512 | correctness + `simd_verification` |

Record new release measurements here after each real-machine validation pass.

## v1.5.x historical summary

v1.5.0 introduced native transcendentals on AVX2, NEON, and AVX-512 and changed `simd_verification` to report per-operation-type geometric means rather than one composite number.

Representative v1.5.x results:

| SIMD | PDF geomean | LogPDF geomean | CDF geomean |
|---|---:|---:|---:|
| AVX2+FMA | 8.0x | 9.6x | 3.3x |
| NEON | 5.9x | 7.3x | 3.1x |
| AVX-512 | 4.8x | 5.1x | 2.2x |

Primitive speedups from the v1.5.x validation cycle:

| SIMD | VectorExp | VectorLog | VectorErf | VectorCos |
|---|---:|---:|---:|---:|
| AVX2+FMA | 3.4x | 1.7x | 2.5x | 4.9x |
| NEON | 2.1x | 1.8x | 8.0x | 3.0x |
| AVX-512 | 5.0x | 3.9x | 1.3x | 8.5x |

## Running benchmarks

Use Release builds for performance numbers:

```bash
cmake -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release --parallel
./build-release/tools/simd_verification
```

Record CPU model, OS, compiler, build type, and relevant SIMD capability output from `system_inspector --quick` with any result set.

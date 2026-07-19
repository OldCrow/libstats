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

## v2.0.2 scipy comparison benchmark

Cross-library throughput and accuracy comparison run via `pylibstats/benchmarks/scipy_comparison.py` (pylibstats v0.3.2, scipy 1.18.0, numpy 2.4.4). Results capture the state after the `perf/dispatch-threshold-recalibration` threshold updates.

### Machines

| Machine | CPU | SIMD | OS | Python |
|---|---|---|---|---|
| Asus TUF A16 | AMD Ryzen 7 7445HS (Zen 4) | AVX-512 | Windows 11 | 3.12.10 |
| MacBook Pro 14,1 (2017) | Intel Core i7-7820HQ (Kaby Lake) | AVX2+FMA | macOS Tahoe | 3.14.6 |

### Peak throughput highlights (pylibstats absolute, at optimal batch size)

| Distribution / Op | Zen4 AVX-512 | Kaby Lake AVX2+FMA | Zen4 peak N |
|---|---|---|---|
| Exponential log_PDF | 2.2G/s | 490M/s | 30k |
| Uniform log_PDF | 1.5G/s | 1.1G/s | 20–30k |
| Gaussian log_PDF | 2.1G/s | 760M/s | 25k |
| Exponential PDF | 993M/s | 303M/s | 30k |
| Uniform PDF | 1.5G/s | 966M/s | 20–30k |

All peak measurements were taken at sizes below the L2→L3 boundary (~50k elements on Zen4; see `SIMD_OPTIMIZATION_REFERENCE.md §Cache hierarchy effects`).

### Throughput ratios vs scipy (selected, at N=100k)

Speedup ratios vary with N. N=100k is representative for sustained throughput above the L3 cache fill threshold on both machines.

| Distribution | Op | Zen4 | Kaby Lake |
|---|---|---|---|
| Exponential | log_PDF | 29× | 11× |
| Gaussian | log_PDF | 6× | 22× |
| Gamma | log_PDF | 18× | 9× |
| VonMises | log_PDF | 25× | 24× |
| StudentT | PDF | 35× | 23× |
| Uniform | PDF | 19× | 25× |
| Cauchy | CDF | 0.2× | 1.1× | |
| VonMises | CDF | 0.1× | 0.1× |
| Binomial | CDF | 0.9× | 0.4× |

Negative ratios (<1×) indicate distributions with structural algorithm limitations; see `SIMD_OPTIMIZATION_REFERENCE.md §Known structural performance ceilings`.

### Accuracy vs scipy (max relative error, N=50k uniform grid)

All values are bit-identical between Zen4 and Kaby Lake **except VonMises PDF/LogPDF**, which differ due to the Bessel function Tier 1/Tier 2 selection:

| Distribution | pdf | log_pdf | cdf |
|---|---|---|---|
| Gaussian | 1.0×10⁻¹⁵ | 4.4×10⁻¹⁶ | 9.7×10⁻¹⁵ |
| Exponential | 2.2×10⁻¹⁶ | 0 | 1.5×10⁻¹⁴ |
| Laplace | 0 | 1.4×10⁻¹³ | 0 |
| Uniform | 0 | 0 | 2.2×10⁻¹⁶ |
| StudentT | 1.5×10⁻¹⁵ | 5.4×10⁻¹⁶ | 2.7×10⁻⁹ |
| Gamma | 2.0×10⁻¹⁵ | 3.3×10⁻¹⁶ | 1.8×10⁻⁹ |
| LogNormal | 7.3×10⁻¹⁵ | 2.7×10⁻¹⁵ | 2.6×10⁻⁷ |
| VonMises (Zen4/MSVC) | 8.4×10⁻¹⁶ | 4.9×10⁻¹¹ | 4.6×10⁻⁶ |
| VonMises (Kaby Lake/AppleClang) | 2.3×10⁻⁹ | 3.3×10⁻⁹ | 4.6×10⁻⁶ |

The StudentT/Gamma CDF errors (~10⁻⁹) and LogNormal CDF error (~10⁻⁷) are consistent between machines and reflect approximation limits in the regularised incomplete-beta and erfc paths, not SIMD errors. See issues #49 and related.

### Benchmark command

```bash
# From the pylibstats repository root:
python benchmarks/scipy_comparison.py --sizes 1000,10000,100000,1000000

# Fine-grained sweep to detect dispatch threshold issues (issue #50 methodology):
python benchmarks/scipy_comparison.py --sizes 1000,5000,10000,20000,30000,40000,50000,60000,70000,80000,90000,100000
```

---

## NEON threshold recalibration sweep

Sweep run via `pylibstats/benchmarks/scipy_comparison.py` (pylibstats v0.3.2, scipy 1.17.1, numpy 2.4.4) to identify and resolve `kNeon` dispatch threshold floor artefacts. Branch: `perf/dispatch-threshold-recalibration`.

### Machine

| Machine | CPU | SIMD | OS | Python |
|---|---|---|---|---|
| MacBook Pro (14,1, 2017) | Apple M1 | NEON | macOS | 3.14.6 |

### Key finding: L1 cache boundary at N ≈ 5k

M1 L1 data cache is 64KB per performance core. Two-array footprint (input + output doubles) exceeds L1 at N ≈ 4096 elements, producing a systematic throughput trough at N ≈ 5k across all operations regardless of threshold. Operations with `kNeon` threshold=64 (profiler floor) have parallel always active; their N=5k troughs are entirely cache-driven. See `SIMD_OPTIMIZATION_REFERENCE.md §L1 data cache boundary on NEON/M1`.

### Threshold corrections

Three operations showed genuine dispatch troughs with minimum at N=10k (above the L1 boundary):

| Distribution / Op | Old kNeon | New kNeon | Evidence |
|---|---|---|---|
| Laplace PDF | 6144 (floor artefact) | 35000 | Trough at N=10k (268M/s vs 330M/s VECTORIZED); parallel entry first exceeds VECTORIZED at N=35k (+20%) |
| Rayleigh PDF | 10000 | 20000 | Trough at N=10k (179M/s vs 214M/s VECTORIZED); recovery at N=15k; 20k conservative safe zone |
| LogNormal PDF | 10000 | 25000 | Trough extends N=10k–15k (180M/s vs 207M/s VECTORIZED); recovery at N=20k; 25k conservative safe zone |

### Benchmark commands

```bash
# From the pylibstats repository root:
# Standard sweep to identify candidates:
python benchmarks/scipy_comparison.py --sizes 1000,5000,10000,20000,30000,40000,50000,60000,70000,80000,90000,100000

# Focused 5k-resolution sweep for trough pinpointing and threshold verification:
python benchmarks/scipy_comparison.py --sizes 1000,5000,10000,15000,20000,25000,30000,35000,40000,45000,50000
```

---

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

## Issue #33 — gather-vs-polynomial exp/log experiment

Evaluates whether a table-lookup + short-polynomial transcendental using
hardware gather can beat the current SLEEF-derived polynomial
`vector_exp`/`vector_log` (see PLAN.md "Issue #33 Experiment" for full gates
and governance). Tool: `tools/gather_throughput_probe.cpp` (opt-in,
`LIBSTATS_BUILD_SIMD_DEV_TOOLS=ON`).

### Kaby Lake AVX2 result (2026-07-18): null result, closed

Machine: Intel Core i7-7820HQ (Kaby Lake), AVX2+FMA. Measured
`_mm256_i32gather_pd` throughput (ns per 4-wide op) against an FMA-only
baseline under three cache regimes:

| Regime | ns/op | vs FMA baseline |
|---|---:|---:|
| FMA-only baseline | 0.33 | 1x |
| Warm (table resident in L1) | 2.30 | 7.0x |
| Interleave (realistic cache pressure, the gate) | 2.81 | 8.6x |
| Cold (clflush before every gather) | 458.9 | 1406x (flush-only: 306 ns) |

Even the best case (warm) costs ~7x a single FMA -- more than the ~7
polynomial terms a 3-term table replacement would save relative to the
current 10-term `vector_exp_avx2`. This is a floor, not a ceiling: index
computation, range reduction, and edge-case handling in a real kernel only
add cost on top of the isolated gather. AVX2 gather-based exp/log does not
beat the current polynomial on this hardware. No further AVX2 work planned;
production kernels unchanged.

### AVX-512 Zen 4 (Asus TUF A16) result (2026-07-18): kill-gate clears for exp

Machine: AMD Ryzen 7 7445HS (Zen 4), AVX-512. Measured `_mm512_i64gather_pd`
(8-wide) throughput against an FMA-only baseline under the same three cache
regimes, alongside the AVX2 (4-wide) path re-measured on this same box for
direct comparison:

| Regime | AVX2 ns/op | AVX2 vs FMA | AVX-512 ns/op | AVX-512 vs FMA |
|---|---:|---:|---:|---:|
| FMA-only baseline | 0.709 | 1x | 0.404 | 1x |
| Warm (table resident in L1) | 0.699 | 0.99x | 0.493 | 1.22x |
| Interleave (realistic cache pressure, the gate) | 1.027 | 1.45x | 0.688 | 1.70x |
| Cold (clflush before every gather) | 248.3 | 350x | 268.3 | 664x (flush-only: 86.7 ns) |

Contrast with the closed Kaby Lake result (AVX2 interleave 8.6x the FMA
baseline): even the AVX2 path on this Zen 4 machine costs only 1.45x, and
native AVX-512 gather costs 1.70x. This confirms AMD's Zen 4 gather unit is
substantially cheaper than Intel's Skylake-derived one, not just
architecturally different.

Gate math (the FMA baseline measures a 3-term Horner chain, ≈2 FMAs, so
treat it as one "3-term-poly unit"): `vector_exp_avx512` is 10-term: a
3-term table replacement saves 7 terms ≈ 2.33 units, comfortably above the
1.70-unit gather cost -- **the kill-gate clears for exp**, the opposite
outcome from Kaby Lake. `vector_log_avx512` is 7-term: a 5-term ARM-glibc-
style replacement saves only 2 terms ≈ 0.67 units, below the 1.70-unit
gather cost on this simple term-count model -- log does not clear on the
same model alone, though real range-reduction savings (not captured by a
term count) could shift this and are not measured here. As with Kaby Lake,
this is a floor, not a ceiling: a real kernel adds index computation, range
reduction, and edge-case handling on top of the isolated gather.

### AVX-512 Zen 4 (Asus TUF A16) Stage 3 result (2026-07-19): null result

The Stage 1-2 kill-gate cleared exp on the cost of a *single* gather, but the
single-gather ARM `exp_advsimd` variant is only ~1.9 ULP and fails libstats'
accuracy floor. Reaching < 1 ULP requires ARM's `tail` correction -- a *second*
gathered value per element. The Stage 3 kernel is a faithful two-gather port of
ARM optimized-routines' scalar `exp` (MIT source, N=128 tail-corrected table,
order-5 polynomial), built as an opt-in dev-tool kernel only; production
`vector_exp_avx512` is untouched.

Accuracy vs a 1018-point mpmath correctly-rounded reference (per issue #46):

| Kernel | core (abs x ≤ 700) max | mean | IEEE edges (±inf/NaN/over/underflow) |
|---|---:|---:|---|
| table-gather exp (experimental) | 1 ULP | 0.001 ULP | correct |
| current polynomial exp | 1 ULP | — | clamps beyond ±708 |

Accuracy gate **PASS** -- the table kernel matches the current kernel's 1 ULP
and is additionally correct at the edges the current kernel clamps.

Throughput (ns per element, AMD Ryzen 7 7445HS, lower is better):

| Regime | current poly | table-gather | speedup |
|---|---:|---:|---:|
| hot (8K elems, cache-resident) | 2.04 | 1.95 | +4.3% |
| stream (256K elems, ~L3, realistic) | 0.55 | 0.99 | −44.5% |

Performance gate **FAIL** (needed ≥20% at the realistic regime). Two 8-wide
gathers cost more than the current 10-term SLEEF polynomial, which touches no
memory; and the current kernel is already memory-bandwidth-bound (~0.55
ns/elem) when streaming, so the extra gather traffic only makes the table
kernel slower.

**Verdict: null result.** The accurate (< 1 ULP) table-exp does not beat the
current polynomial on Zen 4 -- tied at best when cache-resident, ~1.8× slower
under realistic memory pressure. The exp table port is abandoned; production
kernels are unchanged. Methodological note: the Stage 1-2 throughput probe was
necessary but not sufficient -- it modeled a single-gather variant that cannot
meet the accuracy floor, so only the full < 1 ULP kernel could settle the
question. With this, the entire x86 half of Issue #33 Q2 is closed null (AVX2
and AVX-512); only the NEON Q1 path (needs the M1) remains open.

## Running benchmarks

Use Release builds for performance numbers:

```bash
cmake -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release --parallel
./build-release/tools/simd_verification
```

Record CPU model, OS, compiler, build type, and relevant SIMD capability output from `system_inspector --quick` with any result set.

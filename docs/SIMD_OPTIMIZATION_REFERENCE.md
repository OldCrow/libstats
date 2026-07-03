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

## Detecting threshold miscalibration via external benchmarks

`scripts/PROFILING_METHOD.md` documents the authoritative `strategy_profile`-based calibration procedure. External language bindings (e.g. `pylibstats`) provide a complementary and faster signal: run a throughput sweep across a dense size grid and look for throughput troughs.

### The trough-at-threshold signature

When a dispatch threshold is too low, the parallel strategy fires before its overhead is amortised. The benchmark exposes this as a V-shaped throughput trough whose minimum occurs at the threshold value:

- Throughput rises with N up to just below the threshold (pure SIMD).
- At the threshold, throughput drops sharply as threading overhead dominates.
- Throughput recovers as N grows and parallel work amortises the overhead.

Example from the v2.0.2 AVX-512 recalibration (perf/dispatch-threshold-recalibration):

| Distribution / Op | Threshold | Trough N | Trough throughput | Pre-trough |
|---|---|---|---|---|
| Laplace log_PDF | 64 (profiler floor) | 10k | 129M/s | 229M/s at N=5k |
| Laplace log_PDF | 25k (first pass) | 25k | 170M/s | 433M/s at N=20k |
| Laplace log_PDF | 50k (final) | none | — | monotone above N=50k |
| Uniform CDF | 128 | 10k | 109M/s | 833M/s at N=5k |
| Uniform PDF/LogPDF | 50k | 50k | 438M/s | 1.3G/s at N=40k |

The Laplace log_PDF case illustrates the diagnostic loop: a first-pass threshold of 25k moved the trough from N=10k to N=25k rather than eliminating it. A finer sweep (5k resolution) pinpointed that the threshold only amortises at N=45–50k, motivating the final value of 50k.

### Dispatch effect vs cache boundary effect

A cache hierarchy boundary produces a visually similar throughput drop but cannot be eliminated by adjusting thresholds:

- **Dispatch effect**: trough disappears when threshold is set to NEVER.
- **Cache boundary**: trough persists even with NEVER threshold.

Example: Uniform CDF on Zen4 (AVX-512) shows a throughput drop from ~880M/s at N=45k to ~550M/s at N=50k with NEVER threshold. This is the L2→L3 boundary: two-array footprint (input + output) of 45k doubles = 720KB (fits in Zen4 L2), 50k doubles = 800KB (does not). No threshold change can raise throughput above the L2 ceiling at sizes that exceed L2 capacity.

A 50k threshold previously placed parallel overhead on top of a cache miss (463M/s instead of 552M/s). Setting NEVER removes the overhead and exposes the true cache-limited floor.

### Profiler floor artefacts in the threshold table

A threshold of 64 in `kAvx512` or any architecture table is a strong signal that the profiler measurement floor was hit rather than a real crossover being measured. See `scripts/PROFILING_METHOD.md §Timer jitter and the sub-64 measurement floor` for the mechanism. When a 64 threshold produces a trough in an external benchmark, replace it with a measured amortisation point from either a fine-grained sweep or a fresh `strategy_profile --large` run.

In the v2.0.2 recalibration, `kAvx512` Laplace PDF/LogPDF thresholds of 64 were both documented as profiler floor artefacts and confirmed by the external benchmark trough at N=10k.

### NEVER thresholds for trivial SIMD operations

Distributions whose per-element SIMD cost is very low (Uniform, Laplace at small N) may show that parallel never sustains an advantage within practical batch sizes. For these, NEVER is correct regardless of what the profiler reports at its measurement ceiling. Indicators:

- Parallel never recovers to pre-threshold SIMD throughput across a 1k–100k sweep.
- The distribution performs arithmetic or a single transcendental per element with no iterative path.
- The trough-at-threshold depth exceeds 50% throughput loss.

For AVX-512 v2.0.2, Uniform PDF, Uniform LogPDF, and Uniform CDF are all NEVER.

---

## Cache hierarchy effects on batch throughput

AVX-512's higher per-element throughput means the working set grows faster than on AVX2 machines. The L2→L3 and L3→DRAM transitions are therefore more visible in fine-grained throughput sweeps.

Zen4 (Ryzen 7 7445HS) cache topology:
- L2: ~1MB per core (effective user-data capacity ~800KB)
- L3: 16MB shared
- DRAM: DDR5

Observed thresholds in the v2.0.2 benchmark sweep (two-array footprint = input + output):

| Transition | Two-array size | Approx N (doubles) | Typical throughput drop |
|---|---|---|---|
| L2 → L3 | ~800KB | ~50k | 30–60% for compute-bound distributions |
| L3 → DRAM | ~16MB | ~1M | 30–70% for compute-bound distributions |

The L2 boundary is architecture-specific and must not be assumed to hold on AVX2 (Kaby Lake) or NEON (M1). Throughput comparisons across architectures should be made at sizes that fit in each machine's L2 to avoid confounding compute throughput with memory bandwidth.

---

## Cross-architecture accuracy differences

### Bessel function tier selection

`include/core/bessel.h` provides two tiers for `bessel_i0`, `bessel_i1`, and `log_bessel_i0`:

- **Tier 1** (MSVC/GCC/Clang with `LIBSTATS_HAS_CXX17_BESSEL`): delegates to `std::cyl_bessel_i` (C++17 §29.9.3). Achieves <1 ULP against scipy for κ=2.
- **Tier 2** (AppleClang/macOS, `LIBSTATS_HAS_CXX17_BESSEL` not defined): A&S §9.8.1–9.8.4 polynomial approximation. Documented precision: <1.6×10⁻⁷.

The tier selection is automatic: CMakeLists.txt probes for `std::cyl_bessel_i` and defines `LIBSTATS_HAS_CXX17_BESSEL` only when available. AppleClang does not implement C++17 special functions as of Xcode 16.

### Implication for VonMises accuracy

VonMises PDF and LogPDF accuracy differs between MSVC and AppleClang builds because the normalisation constant `log(2π·I₀(κ))` is computed via different Bessel tiers. This shows up as:

| Platform | VonMises pdf max rel err | VonMises log_pdf max rel err |
|---|---|---|
| Zen4 / MSVC (Tier 1) | ~1 ULP (~8×10⁻¹⁶) | ~5×10⁻¹¹ |
| Kaby Lake / AppleClang (Tier 2) | ~2×10⁻⁹ | ~3×10⁻⁹ |

The `vector_cos` implementations between AVX2 and AVX-512 are algorithmically identical (same 7-term FMA Horner polynomial), confirmed by cross-machine analysis. The accuracy difference is entirely in the scalar Bessel normalisation path, not in any SIMD kernel. See issue #47 for the proposed Tier 2 upgrade.

### Scipy version independence

All other distributions (Gaussian, Exponential, Laplace, Gamma, etc.) produce bit-identical or near-identical accuracy results across Zen4 and Kaby Lake with the same scipy version. The VonMises accuracy difference was initially suspected to be a scipy version artefact (1.17.1 vs 1.18.0); upgrading to 1.18.0 on Zen4 left the Zen4 accuracy unchanged, ruling out scipy.

---

## Known structural performance ceilings

These distributions have throughput limitations that are inherent to their algorithms, not to dispatch thresholds or SIMD kernel quality:

### VonMises CDF

The CDF has no closed form. The current implementation is a scalar integration loop. Throughput: ~200–900k elements/second vs ~30–50M/s for scipy (which uses Cephes adaptive quadrature, also scalar, but more efficient). No SIMD kernel or threshold change will address this. See issue #51 for a precomputed table design.

### Cauchy CDF

Cauchy is implemented as a delegation wrapper over StudentT(ν=1). The CDF therefore evaluates the regularised incomplete-beta function rather than the trivial closed form `arctan((x-x₀)/γ)/π + 0.5`. This makes Cauchy CDF 2–5× slower than scipy on Zen4 where scipy vectorises the arctan path. See issue #48.

### Binomial CDF and PDF

Binomial PMF uses lgamma log-space evaluation in a scalar loop. There is no SIMD log-gamma primitive (`vector_lgamma` is deferred). Binomial CDF uses PMF summation rather than the regularised incomplete-beta approach used by scipy. Both paths cap throughput at ~5–16M elements/second regardless of batch size or architecture. See issue #52.

---

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

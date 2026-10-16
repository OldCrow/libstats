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
- v2.4.0: all four threshold tables (kAvx512/kAvx2/kAvx/kNeon) re-measured
  with sustained crossovers after the parallelForSlices fork repair (#143);
  kAvx measured for the first time via a `LIBSTATS_MAX_SIMD_TIER=AVX`
  capped build
- v2.5.0: `vector_log1p` and `vector_expm1` (`src/simd_dispatch.cpp`), built
  on the dispatched `vector_log`/`vector_exp`, so every tier has them; kept
  only where they win (Gamma, Beta, Pareto, Weibull batch paths). von Mises
  batch PDF/LogPDF use `vector_sin`. clang-cl builds the Windows performance
  path with cl.exe's `/arch:` flags; see "MSVC vectorization limits" below.
- v2.5.0 dispatch: below ParallelUtils' fork threshold PARALLEL runs the
  batch (VECTORIZED) kernel, not a serial scalar loop (ee11e62), and
  `parallelForSlices` folds a partial last slice into the one before it, so
  every strategy returns VECTORIZED's bits (045acd9, #191). On 2026-10-14
  `scripts/derive_threshold_rows.py` re-derived all four tables from the R8
  three-run bundles at code head d1adc95 (`data/profiles/dispatcher/`); a
  back-to-back A/B with `tools/bench/table_ab_bench.cpp` then settled the kNeon,
  kAvx512, kAvx2 and kAvx rows (580b7f7, bb9ac0b, 6c149f1, 4c3a133). Evidence and numbers:
  `SIMD_BENCHMARK_RESULTS.md` §v2.5.0.

Which `simd_*.cpp` files compile for each tier, and how
`LIBSTATS_MAX_SIMD_TIER` caps them, is in `cmake/SIMDDetection.cmake`
(`apply_simd_tier_cap`).

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

### L1 data cache boundary on NEON/M1

The M1 performance core has a 64KB L1 data cache. The two-array footprint (input + output doubles) crosses the L1→L2 boundary at N ≈ 64KB ÷ 16 bytes = 4096 elements. Throughput sweeps on M1 show a systematic drop at N ≈ 5k regardless of dispatch threshold — this mimics a dispatch trough but is cache-driven and cannot be fixed by raising thresholds.

Two indicators distinguish L1 cache effects from dispatch troughs:

- **Trough position**: L1 boundary troughs are pinned at N ≈ 5k on M1 regardless of threshold. A genuine dispatch trough minimum shifts to match the threshold value.
- **NEVER test**: set the threshold to NEVER and re-sweep. A dispatch trough disappears; an L1 cache trough persists.

Up to v2.4.x, operations with `kNeon` threshold=64 (profiler floor artefact) had parallel always active, so their N=5k throughput drop was entirely the L1→L2 transition and did not warrant recalibration. See `SIMD_BENCHMARK_RESULTS.md §NEON threshold recalibration sweep` for a worked example. Since ee11e62, PARALLEL below the fork threshold runs the VECTORIZED kernel, and since 2026-10-14 no table holds a value below 2048.

### NEVER thresholds for trivial SIMD operations

Distributions whose per-element SIMD cost is very low (Uniform, Laplace at small N) may show that parallel never sustains an advantage within practical batch sizes. For these, NEVER is correct regardless of what the profiler reports at its measurement ceiling. Indicators:

- Parallel never recovers to pre-threshold SIMD throughput across a 1k–100k sweep.
- The distribution performs arithmetic or a single transcendental per element with no iterative path.
- The trough-at-threshold depth exceeds 50% throughput loss.

For AVX-512 v2.0.2, Uniform PDF, Uniform LogPDF, and Uniform CDF were all NEVER; the 2026-10-14 `kAvx512` rows are 300k/300k/100k. Of the Uniform rows, only `kNeon` PDF remains NEVER (`kAvx2` CDF took 25k in 4c3a133).

### The secretly-serial parallel path signature (v2.4.0)

If forced-PARALLEL timings are **bitwise identical to VECTORIZED at every
batch size**, the parallel path never actually forked — every threshold
derived from that data calibrates a serial path and is invalid. The v2.4.0
instance (#143): 18 sliced batch sites passed slice COUNTS (~N/1024) to
element-denominated `parallelFor` gates, so PARALLEL ran serial below
~8.4M elements, and the June tables encoded exactly that. Check for this
signature FIRST when profiling: a genuine parallel path differs from
VECTORIZED at least in noise.

### NEVER for memory-bound kernels (v2.4.0)

Distinct from the trivial-op class above: a kernel can be expensive enough
to vectorise well and still lose under a REAL fork because it is
memory-bound — the fork adds coordination without adding usable bandwidth.
In the v2.4.0 measurement Beta PDF/LogPDF ran 0.65–0.92× at 2M on every
tier (AVX-512, AVX2, capped AVX, NEON), and von Mises batch CDF fell in the
same class (#111/#144); both were encoded NEVER. The 2026-10-14
re-derivation took them off NEVER: Beta PDF/LogPDF are 8192/10000 on
`kNeon`, 4096/25000 on `kAvx`, 4096/8192 on `kAvx2` and 8192/8192 on
`kAvx512`, and the quadrature CDF (96b157f) has no full-array sweep left
(see "VonMises CDF" below). The diagnosis still holds for a kernel that
measures below 1× at every size. Corollary: per-tier kernel economics can
flip the verdict for compute-bound kernels — in v2.4.0 HalfNormal CDF was
NEVER on NEON (2.2 ns/elem `vector_erf`, nothing left for 8 cores to win)
but 1.3–2.1× parallel on x86, whose `vector_erf` is ~5× slower per element
(corvus#37); its 2026-10-14 `kNeon` row is 75000.

### Extraction rule: sustained crossovers (v2.4.0)

Read profiles with the sustained V→P crossover — parallel wins at that
size AND every larger size — not the first crossing, which reports
64-element noise ties. Binding statement: the 2026-09-04 amendment in
`scripts/PROFILING_METHOD.md`. `scripts/analyze_crossovers.py` computes it
(#146); `scripts/derive_threshold_rows.py` builds on it to turn bundles into
table rows.

### Clean-entry criterion for iterative calibration

When iterating threshold values, the target state is a parallel entry that exceeds the last VECTORIZED point in the sweep. If the first benchmark measurement at the new threshold T falls below the VECTORIZED level just below T, the parallel path has not yet amortised its threading overhead — raise the threshold further.

Example from the NEON Laplace PDF recalibration (perf/dispatch-threshold-recalibration, M1):

| Threshold | Last VECTORIZED | Parallel entry (N=T) | Entry delta |
|---|---|---|---|
| 25k | 327M/s at N=20k | 316M/s | −3% (parallel losing — raise threshold) |
| 30k | 385M/s at N=25k | 363M/s | −6% (parallel losing — raise threshold) |
| 35k | 383M/s at N=30k | 460M/s | +20% (parallel winning — correct threshold) |

At 25k and 30k the parallel entry was below VECTORIZED, meaning the threshold fired before overhead amortised. At 35k parallel immediately exceeded VECTORIZED; 35k is the encoded value.

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

`include/libstats/core/bessel.h` provides two tiers for `bessel_i0`, `bessel_i1`, and `log_bessel_i0`:

- **Tier 1** (MSVC/GCC/Clang with `LIBSTATS_HAS_CXX17_BESSEL`): delegates to `std::cyl_bessel_i` (C++17 §29.9.3). Achieves <1 ULP against scipy for κ=2.
- **Tier 2** (AppleClang/macOS, `LIBSTATS_HAS_CXX17_BESSEL` not defined): A&S §9.8.1–9.8.4 polynomial approximation. Documented precision: <1.6×10⁻⁷.

The tier selection is automatic: CMakeLists.txt probes for `std::cyl_bessel_i` and records `LIBSTATS_HAS_CXX17_BESSEL` in the generated `libstats_config.h` (#97), so the library, tests and consumers compile the same tier. AppleClang's libc++ does not implement the C++17 special functions, so macOS builds take Tier 2. Since 8d3c9c7, Tier 1 `bessel_i0` returns 1 below 1e-8 (libstdc++'s `std::cyl_bessel_i(0, 5e-324)` is NaN), and `log_bessel_i0` and the tier-independent `log_bessel_i0_scaled` split log(2πx) past DBL_MAX/2π instead of returning −inf. The tier-independent A(κ) = I₁/I₀ ratio helpers build on whichever tier is active.

### Implication for VonMises accuracy

Up to v2.4.1, VonMises PDF and LogPDF accuracy differed between MSVC and
AppleClang builds because the normalisation constant `log(2π·I₀(κ))` came from
different Bessel tiers (~1 ULP on Tier 1, ~2×10⁻⁹–3×10⁻⁹ on Tier 2 at the
time).

From v2.5.0 the density no longer depends on the tier. Scalar and batch
PDF/LogPDF evaluate log f = −2κ·sin²(t/2) − log(2π·I₀(κ)·e^−κ), t = x − μ,
without the κ·cos t − log I₀(κ) cancellation. The scaled normaliser
I₀(κ)·e^−κ comes from a trapezoid rule on its periodic integral up to κ = 1000
and from the Hankel series above (`src/von_mises.cpp`, `updateCacheUnsafe`).
The batch kernel is `scalar_add(−μ)`, `scalar_multiply(½)`, `vector_sin`,
`vector_multiply` (square), `scalar_multiply(−2κ)`,
`scalar_add(−log scaled normaliser)`. The Bessel helpers in `bessel.h` still
serve the circular moments. Per-ISA figures are in the generated blocks of
`docs/ACCURACY_CHARACTERIZATION.md`; the AVX-512, AVX2 and NEON blocks give
identical von Mises rows (pdf max_rel 1.724e-12, logpdf 7.312e-15).

The `vector_cos`/`vector_sin` implementations are the clean-room quadrant-reduction kernel of #95 on every tier (shared `trig_cleanroom_data.inc`, max 1 ULP), so AVX2 and AVX-512 remain algorithmically identical.

### Scipy version independence

All other distributions (Gaussian, Exponential, Laplace, Gamma, etc.) produce bit-identical or near-identical accuracy results across Zen4 and Kaby Lake with the same scipy version. The VonMises accuracy difference was initially suspected to be a scipy version artefact (1.17.1 vs 1.18.0); upgrading to 1.18.0 on Zen4 left the Zen4 accuracy unchanged, ruling out scipy.

---

## Known structural performance ceilings

These distributions have throughput limitations that are inherent to their algorithms, not to dispatch thresholds or SIMD kernel quality:

### VonMises CDF

The CDF has no closed form. Historically a scalar integration loop (~200–900k elem/s vs scipy's ~30–50M/s Cephes quadrature — the numbers in the tables above date from that era). v2.3.0 (#51) shipped a cached Bessel-series CDF for κ ≤ 1000 with a wrapped-normal fallback above, whose batch path was memory-bound (#111).

From v2.5.0 (96b157f, closing #111) the CDF is a quadrature in w = κ(1 − cos θ): a short Gauss–Legendre rule for the central mass within ¼ of the median, and beyond it the tail mass by a Watson-lemma series or Gauss–Legendre (`src/von_mises.cpp`). It is exact to κ = 1e20; above that it takes the normal limit Φ(2√κ·sin(t/2)). The Bessel series and the wrapped-normal fallback are gone. The batch CDF evaluates per element on one snapshot, at a fixed cost per element: one central-mass rule (6–12 `exp`) or one tail evaluation. Its 2026-10-14 dispatch rows are 2048 (`kNeon`), 4096 (`kAvx`, `kAvx2`) and 8192 (`kAvx512`).

### Cauchy CDF

Only Cauchy LogPDF (and sampling) delegate to StudentT(ν=1); the PDF is its own kernel, and the CDF has been the closed-form arctan since #48 (v2.3.0) — one `std::atan` per element, ~2 ULP. The sub-unity scipy ratios in the tables above predate that change; #109 (closed 2026-09-04) retired the matching stale dispatch rows with measured values.

### Binomial CDF and PDF

Binomial PMF evaluates `detail::binomial_log_pmf` per element in a scalar loop, without the n·log n cancellation of the lgamma form (#172). There is no SIMD log-gamma primitive (`vector_lgamma` is deferred). Binomial CDF is the regularised incomplete beta, `detail::beta_i_pair` (`src/binomial.cpp`), as in scipy. Both paths are scalar per element; the ~5–16M elements/second ceiling measured for the older PMF-summation CDF (#52) predates both changes.

---

## MSVC vectorization limits

Evidence: `docs/bench-evidence/2026-10-05-zen4-msvc-vs-clangcl/README.md`
(Zen 4, MSVC 19.51 against clang-cl 22.1.3). The Windows benchmark and R3/R8
captures (`kAvx512`, and Zen 4's R3 and R8) come from clang-cl; cl.exe is the
correctness build.

- **No loop around an inner loop.** cl.exe does not vectorize a loop whose
  body contains another loop (C5002, reason 1106). `log1pmx_series`' Horner
  loop over a coefficient array kept the Beta and Gamma Stirling batch
  densities scalar under MSVC, 2.9–4.0× slower per element than clang-cl at
  n = 1e4, until 998d985
  wrote the Horner out (`include/libstats/core/math_utils.h`). Clang unrolls
  the inner loop and vectorizes either form.
- **No select it cannot blend.** `vector_expm1`'s |x| < ½ select gives reason
  1100 in every form tried, so the Pareto and Weibull batch CDFs stay scalar
  under MSVC.
- **No FMA contraction.** MSVC emits no FMAs from `a*b + c`; clang-cl
  contracts by default (111 FMAs in the Beta batch code, 47 in Gamma), so
  clang-cl is faster on the polynomial-heavy rows. The FP-contraction rule in
  `AGENTS.md` covers the accuracy side.

To check a loop, build the TU with `/Qvec-report:2` and read the C5001
(vectorized) and C5002 (not vectorized, with reason) lines; the reason codes
are in Microsoft's "Vectorizer and parallelizer messages" page.

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

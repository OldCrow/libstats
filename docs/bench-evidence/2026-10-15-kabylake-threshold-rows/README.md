# K table A/B on kAvx2 and kAvx, 2026-10-15

Kaby Lake i7-7820HQ (4C/8T), AppleClang 15, macOS 13.7.8. The procedure is PLAN.md R8,
"Table A/B on K and Z":

- Release `libstats_static` builds of two trees, tests and tools off.
- `tools/bench/table_ab_bench.cpp` built from the head tree against each.
- Old, new, old, new back to back per tier, each run started once the 1-min load was below
  2.0, launched detached. The driver scripts and logs are in each `round*/` directory.
- Analysis with M's `table_ab_analyze.py`, its input path read from `$R`.

Tiers: native AVX2 (kAvx2), capped AVX and capped SSE2 (both kAvx, since SSE2 delegates to
kAvx). K's fork threshold is 4096 on every tree.

## WORK_STEALING on K

K is the only machine where AUTO goes multi-threaded through WORK_STEALING (hyper-threaded
macOS). The R8 captures and the derived rows measure PARALLEL. On K, WORK_STEALING wins well
before PARALLEL. For example, Rayleigh LogPDF at 8192, in ns/element: WORK_STEALING 3.06,
VECTORIZED 4.51, PARALLEL 6.82. kAvx2 and kAvx also drive Linux and Windows AVX2/AVX machines,
whose AUTO picks PARALLEL, so the rows stay PARALLEL-optimal, with one exception: where
PARALLEL is break-even (Gaussian LogPDF, below) [user, 2026-10-15].

## Round 1: `580b7f7`'s rows against the header before them (`10a7ff7`)

| Tier | Noise floor (median, p05–p95) | AUTO faster / slower > 1.25× | AUTO > 1.25× behind best: old → new |
|---|---|---|---|
| AVX2 | 1.002 (0.94–1.08) | 156 / 5 | 158 → 15 |
| AVX | 1.010 (0.95–1.11) | 151 / 8 | 146 → 9 |
| SSE2 | 1.000 (0.92–1.06) | 151 / 0 | 173 → 26 |

Table-caused regressions, kAvx2 only, were all where the old table let K's WORK_STEALING run:
Discrete CDF 25k, Rayleigh LogPDF 8192/10k, Laplace PDF 8192/10k. The AVX rows are not
table-caused: AUTO takes the same strategy class under both tables.

Decided [user, 2026-10-15]:
- Keep the new values. Discrete CDF goes to 25k, as a compromise between K and the PARALLEL
  platforms.
- Drop the overrides that the bench shows wrong under both tables (`DROP_OVERRIDE`):
  - kAvx2: Uniform CDF, Exponential PDF and CDF, Gaussian LogPDF.
  - kAvx: Pareto CDF, Rayleigh CDF, Poisson LogPDF.
- Take the one-step candidates (`TAKE_CANDIDATE`):
  - kAvx2: HalfNormal PDF, LogNormal PDF, Gumbel PDF, Fisher-F LogPDF.
  - kAvx: von Mises LogPDF.
- Where the rule's value would itself regress, encode the bench crossover (`BENCH_CROSSOVER`):
  - Exponential CDF (kAvx2) 25k, not 8192: PARALLEL is 1.4–1.5× behind at 8192–10k.
  - Rayleigh CDF (kAvx) 8192, not 4096: PARALLEL is 1.09–1.38× behind at 4096.
- Leave the SSE2-only gaps: kAvx also serves real AVX machines, and kSse2 is a v3 input.

## Round 2: those 13 values against round 1's new header

| Tier | Noise floor | AUTO faster / slower > 1.25× | gaps old → new |
|---|---|---|---|
| AVX2 | 0.997 (0.93–1.08) | 15 / 1 | 15 → 1 |
| AVX | 1.001 (0.94–1.07) | 9 / 0 | 8 → 3 |
| SSE2 | 1.003 (0.91–1.08) | 15 / 0 | 29 → 18 |

One regression: Gaussian LogPDF at 100k (kAvx2, 150k). AUTO stayed VECTORIZED where K's
WORK_STEALING was 2.2× faster; PARALLEL/VECTORIZED at 100k was 1.10 median (0.97–1.20) over
eight passes. One cross-tier gap remained: Uniform PDF at 500k, about 1.9× on every tier.
The 2000k candidate is the maximum of a 10× spread.

Decided [user, 2026-10-15] (`BENCH_CROSSOVER`):
- Gaussian LogPDF (kAvx2) goes to 100k.
- Uniform PDF (kAvx2 and kAvx) goes to 250k, the median profile run. PARALLEL/VECTORIZED
  is 0.54–0.57 at 500k on every tier, and 0.80–0.88 median at 200k with passes up to 1.23.

## Round 3: those three rows against round 2's header

| Tier | Noise floor | AUTO faster / slower > 1.25× | gaps old → new |
|---|---|---|---|
| AVX2 | 0.989 (0.91–1.06) | 5 / 1 | 3 → 2 |
| AVX | 0.998 (0.93–1.06) | 2 / 2 | 2 → 2 |
| SSE2 | 1.001 (0.94–1.06) | 1 / 0 | 20 → 17 |

The Uniform PDF 500k and Gaussian LogPDF 100k gaps closed. The "slower" rows (AVX2 Gaussian
LogPDF 500k, AVX Weibull CDF 8192, Logistic CDF 100k) run multi-threaded under both tables:
they are noise, not the table. What remains:
- AVX's Uniform PDF at 200k (1.26×), by design: 250k keeps VECTORIZED there.
- Two unchanged AVX2 rows at 25k, Cauchy LogPDF and TruncatedNormal PDF, at 1.34–1.37×.
- The SSE2-only rows.

## Final header and R1

Against `31d3ec3`, 15 values change on 13 rows: kAvx2 10 values on 8 rows, kAvx 5 on 5. `rows_kAvx2.txt` and
`rows_kAvx.txt` list every row and its decision; `dispatch_thresholds.diff` is the change.
Rerunning `scripts/derive_threshold_rows.py` on the final header reproduces its values. The
comments on the changed rows were rewritten where they argued for the old values: Exponential,
Gaussian and Uniform in kAvx2. So were the von Mises "#111" notes in kNeon, kAvx, kAvx2 and
kAvx512: #111 closed with the quadrature CDF (`96b157f`), the last #211 item. clang-format
20.1.8 ran on the changed lines only.

R1 (Release): 0 warnings (34 translation units rebuilt), correctness 132/132. Quiet
`test_parallel_batch_gates`: PARALLEL/VECTORIZED 0.270–0.296. Timing 14/14. Logs in `r1/`.

# kAvx on native AVX silicon: derivation and table A/B, 2026-11-21

**Status: the committed table is provisional** [user, 2026-11-21]. See "After the commit" at the
end: sustained timing does not confirm the rows moved into the 4096–8192 zone, and the A/B below
was timed inside the CPU-frequency ramp (#320).

Ivy Bridge i7-3820QM (4C/8T; AVX, no AVX2, no FMA), Linux Mint 22.3, glibc 2.39, GCC 13.3,
`dev/v3.0.0-corvus-adoption` (corvus v1.2.0). The fleet's only machine whose native tier is AVX.
`kAvx` takes this machine's values [user, 2026-11-19]; until now the table came from a Kaby Lake
build capped with `LIBSTATS_MAX_SIMD_TIER=AVX` (#291).

## Capture

`data/profiles/dispatcher/2026-11-21T21-09-06Z_linux-x86_64_dev-v3.0.0-corvus-adoption_sha-cfbb992/`:
three quiet `strategy_profile --large` runs, code `e890dec`. Its manifest has the spread and the
capture conditions.

## Derivation

`scripts/derive_threshold_rows.py --tables kAvx`, the new bundle against the capped Kaby Lake
bundle the previous table came from (`2026-10-13T13-58-00Z_…`). Changes to the script's kAvx
entry, all for this machine:

- **The fork zone ends at the fork value, 4096.** On the capped Kaby Lake build a candidate of
  6144 was taken to mean "PARALLEL wins as soon as it forks" and encoded 4096. Here PARALLEL is
  behind VECTORIZED at 4096 (median of the three runs) on 37 of 81 rows, so a candidate of 6144
  is a measured crossover and is encoded as 6144.
- **The Kaby Lake kAvx entries are gone** from `DROP_OVERRIDE`, `TAKE_CANDIDATE` and
  `BENCH_CROSSOVER` (Pareto CDF, Rayleigh CDF, Poisson LogPDF, von Mises LogPDF, Uniform PDF):
  they recorded that machine's A/B.
- **No Zen 4 direction check**: its capped AVX build has FMA and AVX2 hardware under the cap.

`rows_kAvx.txt` has every row and its decision; `summary.txt` the counts; `dispatch_thresholds.diff`
the header change. 27 values across 19 distributions change, every one earlier than before: without FMA the
VECTORIZED kernels cost more per element, so PARALLEL pays off sooner.

## Round 1: the rule's 30 values against the header before them

`tools/bench/table_ab_bench.cpp` built against Release `libstats.a` of two trees that differ only
in `dispatch_thresholds.h`, run old, new, old, new (`round1/run.txt`). **Loaded**: a desktop
session was in use, 1-minute load 1.4–4.9. `round1/rows_kAvx_proposed.txt` is the proposal that
was tested; `round1/ab_compare.txt` is M's `table_ab_analyze.py` over the four CSVs.

Noise: on the 510 AUTO cells of rows that did not change, new/old is 0.999 median, 0.76–1.24
(p05–p95). So the whole-table listing in `ab_compare.txt` (AUTO more than 1.25× behind its best
forced strategy: 154 → 118) mixes noise with the table's effect, and each changed row was judged
on its own, at the bench sizes between its old and new value, by forced PARALLEL against forced
VECTORIZED in all four files. The bench grid has no 6144 point; a row moving to 6144 is judged
at 8192 and 10000.

26 of the 30 hold: PARALLEL/VECTORIZED time below 1 in every file at every affected size.

| Rows | PARALLEL / VECTORIZED | AUTO new / old |
|---|---|---|
| ChiSquared, Erlang, Gamma, StudentT, von Mises LogPDF; StudentT, TruncatedNormal, ChiSquared, von Mises PDF; Beta LogPDF | 0.20–0.53 | 0.25–0.70 (five cells 0.92–1.01) |
| Laplace, LogNormal, Pareto PDF; Rayleigh, LogNormal, Pareto, Weibull LogPDF; Gumbel, Logistic CDF | 0.56–0.82 | 0.58–0.90 (LogNormal PDF at 10000: 1.07) |
| Cauchy and Gumbel LogPDF at 8192–10000; Weibull LogPDF at 4096 | 0.80–0.94 | 0.73–0.92 |
| Exponential, Gaussian, HalfNormal, Uniform LogPDF; Uniform PDF (the 50k and 75k rows) | 0.32–0.77 | 0.43–0.74 |

Four do not [user, 2026-11-21]:

| Row | Rule's value | Bench | Encoded |
|---|---|---|---|
| Exponential CDF | 25k → 4096 | PARALLEL 1.17–1.35× slower at 4096; 0.73–0.79 at 8192 | 8192 (`BENCH_CROSSOVER`) |
| Rayleigh CDF | 8192 → 4096 | 0.95–1.12 at 4096 (median 1.01) | 8192 kept (`KEEP_ROW`) |
| InverseGamma LogPDF | 25k → 8192 | 1.07–1.14× slower at 8192 and 10000 | 25k kept (`KEEP_ROW`) |
| InverseGamma PDF | 200k → 8192 | 1.02–1.05 at 8192, 0.99 at 10000, 0.86–0.94 at 25k–100k | 200k kept (`KEEP_ROW`) |

The table as committed therefore differs from the one benched in these four rows. Three keep the
old value and one takes a size the bench measured, so no second round was run.

## InverseGamma PDF under AUTO

Under both tables, wherever AUTO goes multi-threaded for InverseGamma PDF it is slower than
either forced strategy (ns/element):

| n | VECTORIZED | PARALLEL | AUTO, old table | AUTO, benched table |
|---|---|---|---|---|
| 8192 | 70 | 71–73 | 68–73 | 127–137 |
| 25000 | 70 | 65 | 113–117 | 117–118 |
| 200000 | 71 | 60 | 117–142 | 109 |

At 25000 the old table's InverseGamma PDF row is 200k, so AUTO should have stayed VECTORIZED;
the PDF delegates to Gamma, whose own row is 6144, and Gamma's row decides.

**Cause, found after the commit: the bench, not the library.** The AUTO PDF cell is timed
straight after a forced-PARALLEL CDF cell. While that runs the calling thread sleeps and its
core's clock drops (2.1–2.8 GHz against 3.5, `schedutil`); InverseGamma PDF does over 90% of its
work serially on the caller (#321), so it is the row that shows it. Timed per block, AUTO PDF
goes 183 → 103 ns/element after a PARALLEL CDF and reads 66–72 after a VECTORIZED one. #320.

## After the commit: a sustained check

Forced VECTORIZED against forced PARALLEL, every row, sizes 4096–500000, each cell warmed for
0.15 s and timed for 0.25 s; 1-minute load 0.9–1.4. Indicative, not a record.

- From 25000 up PARALLEL takes 0.3–0.7× VECTORIZED's time on essentially every row (Cauchy
  LogPDF is the exception: 1.2–1.8× slower at 50k–500k).
- Between 4096 and 10000 the ratio is not monotone: typically good at 8192, bad at 6144 and
  10000 (Poisson LogPDF, a row this commit did not touch: 0.81, 1.94, 0.59, 1.81). Tasks are one
  per 1024-element slice on a four-thread pool, so wall time goes as ceil(slices / 4).
- Moved rows it does not support, PARALLEL / VECTORIZED at 8192 and 10000: Weibull LogPDF 1.36,
  1.89; Beta LogPDF 0.60, 1.86; Erlang LogPDF 0.77, 1.38; ChiSquared LogPDF 0.92, 1.16; Pareto
  LogPDF 1.23, 0.93; Rayleigh LogPDF 1.11, 0.97.
- The A/B above read several forced-VECTORIZED LogPDF cells at 16–17 ns/element where the quiet
  capture and this run read about 9, so its PARALLEL margins on those rows are overstated.

The table stays as committed on the dev branch and is re-captured once `strategy_profile` warms
up each cell (#320).

## Not a quiet record

A quiet repeat is scripted locally (`build-ivb-adopt/ab-kavx/quiet_ab.sh`, three passes) and was
not run: the four doubtful rows read the same way in all four files.

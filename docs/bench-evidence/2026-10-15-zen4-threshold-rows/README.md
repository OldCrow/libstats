# Zen 4 table A/B at d1adc95: kAvx512 (2026-10-15)

Z's run of PLAN.md R8 "Table A/B on K and Z": AUTO under the header before the rows (`10a7ff7`)
against AUTO under the proposed header, with `tools/bench/table_ab_bench.cpp` (27 distributions ×
pdf/logpdf/cdf × 10 sizes, AUTO and forced VECTORIZED/PARALLEL), ABAB, every bench process opted
out of power throttling, a discarded warm-up pass per tier, the clock probe at 127.2–129.8 ms per
pass (full boost) before each tier. Five rounds; rounds 1, 2, 4 and 5 quiet (2.7–8.8% CPU, screen
locked), round 3 indicative (the quiet gate timed out at 11–18%: the .NET runtime install's ngen
queue resumes whenever the machine idles). Outcome: 21 kAvx512 values on 21 rows; AUTO rows more
than 1.25× behind their best forced strategy 208 → 5 (M's kNeon: 202 → 24).

## Builds

- Old: detached worktree `../libstats-oldtable` at `10a7ff7`, clang-cl 22.1.3 (VS 2026 18.11
  bundled), Ninja, Release, static library only, one tree per tier (uncapped `/arch:AVX512`;
  `LIBSTATS_MAX_SIMD_TIER=` AVX2, AVX, SSE2). The two trees differ only in
  `dispatch_thresholds.h` (`git diff --stat 10a7ff7 HEAD -- include src`).
- New: this worktree's `build-clangcl*/` rebuilt at each header under test. 0 warnings everywhere.
- Benches: `table_ab_bench.cpp` from the head tree, built with clang-cl and the tier's `/arch:`
  against each tree's `stats_static.lib` and include directory.

## Round 1 (`round1/`): `122aa51`'s header (the `580b7f7` rows) against `10a7ff7`, four tiers

Each tier started at 2.8–4.3% CPU. Noise floor (forced rows, same code) p05–p95 0.90–1.14 on
AVX-512, tighter on the capped tiers.

AVX-512 (kAvx512): gaps 207 → 103; 112 of 810 AUTO rows faster by more than 1.25×, 22 slower. Every
"slower" row is a single-pass outlier on an unchanged row (Gamma pdf/logpdf at 8192, Discrete and
Exponential/Gaussian logpdf at 500k: one pass 2× the other three, forced strategies unchanged), a
trivial-kernel row at 1k–4k (Uniform, 0.13 vs 0.17 ns), or InverseGamma at 4k–10k where all four
strategies shifted together: no table-caused regression. The 103 remaining gaps sit on the sixteen
rows the derivation kept as standing overrides (`KEEP override` in
`2026-10-14-m1-threshold-rows/rows_kAvx512.txt`) and their delegates: the same gap under both
tables, PARALLEL 0.24–0.35× VECTORIZED from 8192 on the discrete log-pmfs, Student-t and von Mises
CDFs, Beta logpdf, and the Cauchy, Pareto, Rayleigh, Exponential and Gaussian CDFs. Two overrides
the bench supports stay: Fisher-F PDF 8192 (PARALLEL ties VECTORIZED at 4096, 0.40× at 8192) and
InverseGamma PDF 25k (PARALLEL within 1.1× of VECTORIZED through 50k, ahead from 100k).

Capped tiers, for K's tables (Z supplies direction only): kAvx2 196 → 46 gaps, 148 faster, 6 slower
(Uniform 8k–25k and Rayleigh/Exponential logpdf at 4k–8k, all within 1.3× on sub-ns kernels); kAvx on
the AVX tree 164 → 31, 138 faster, 2 slower (Gaussian pdf 25k 1.25×, logpdf 100k 1.30×); kAvx on the
SSE2 tree 198 → 64, 130 faster, 4 slower (Gaussian logpdf 25k–50k 1.3–1.4×). No row regresses by
more than the forced-strategy shift on any capped tier.

## Round 2 (`round2/`): fourteen overrides dropped (procedure step 4)

`DROP_OVERRIDE` in `scripts/derive_threshold_rows.py` gains fourteen kAvx512 rows with the round-1
evidence (an override the bench shows wrong under both tables): Binomial LogPDF, NegBinomial
PDF/LogPDF, Student-t CDF, von Mises CDF, Gaussian CDF → 8192 (the fork value; the candidates sit
in the fork zone); Cauchy CDF, Beta LogPDF, Pareto CDF, Rayleigh CDF, Exponential CDF → 25k (their
candidates); Bernoulli PDF/LogPDF and Geometric PDF → 8192 as the pass-through mirrors of Binomial's
and NegBinomial's rows (the bench shows Bernoulli and Geometric with their delegates' gaps).

Quiet (4.5%). Gaps 209 → 18; 194 AUTO rows faster, 7 slower (Uniform and Exponential at 1k–2k,
0.1–0.3 ns/element kernels where AUTO and the best forced strategy swap within noise under both
tables; M's "not a table matter" class).

## Round 3 (`round3/`): three one-step rows taken [user]

`TAKE_CANDIDATE`: Logistic LogPDF 50k → 25k, Gumbel CDF 50k → 25k, TruncatedNormal LogPDF
75k → 50k (candidates measured three times; PARALLEL 0.57–0.73× VECTORIZED at the candidate in
rounds 1–2, AUTO 1.37–1.75× behind). Indicative only (11–18% CPU): the three rows reach the best
strategy's speed; the rest of the output is load. Rerun quiet as round 4.

## Round 4 (`round4/`): the round-3 header, quiet (3.3–6.6%)

Gaps 211 → 19; 193 faster, 3 slower (Uniform and Discrete at 50k, 1.26–1.27×, inside the noise
band). The 19 remaining are the fork-zone class: rows whose profile candidate was the max of a
> 2× spread (25k) while the loaded A/B shows PARALLEL ahead at 8192 in every pass of every quiet
round — Beta PDF (0.51–0.65×), Beta LogPDF (0.54–0.74×), Pareto CDF (0.57–0.61×), Cauchy CDF
(0.65×) — plus Gamma at 8192–10k (below) and trivial-kernel rows.

## Round 5 (`round5/`): the four fork-zone rows to 8192, quiet (3–5%)

New rule `BENCH_CROSSOVER` in the script (user's decision, evidence per row): Beta PDF/LogPDF,
Pareto CDF, Cauchy CDF 25k → 8192. Gaps 208 → 5; 199 faster, 2 slower (ChiSquared logpdf 8192 and
Uniform cdf 25k, 1.26–1.27×, noise band). The four rows run AUTO at the best strategy's speed
(1.00–1.08×). The five remaining: Uniform logpdf/cdf and Discrete cdf at 1k–2k (trivial kernels),
Laplace logpdf 10k and LogNormal logpdf 25k at 1.27–1.40× under both tables (unchanged rows, one
grid step inside the rules' band). Conclusive; this is the header for R1.

## The Gamma 8192 observation (`gamma_probe/`)

Rounds 1, 2 and 4 showed Gamma pdf/logpdf under AUTO at 8192–10k reading 4–5 ns against 2.0 forced,
in some passes under both tables, with the Gamma rows unchanged. `gamma_probe.cpp` (built against the
head tree, run quiet, opted out): `PerformanceDispatcher::selectStrategy` returns VECTORIZED for
Gamma pdf/logpdf below 25k and PARALLEL from 25k, as the table says; AUTO at 4096–16384 times
2.01–2.08 ns, identical to forced VECTORIZED, in all three measurement orders × 3 repetitions, and
identical right after a 500k-element PARALLEL call and after a 300 ms pause. The occasional 1.5×
slow sample (3.0 ns) lands on forced VECTORIZED and PARALLEL as often as on AUTO. So the bench's
Gamma rows are measurement outliers (a slow window lasting one or two consecutive measurements),
not a dispatch defect; round 5 shows none. No issue.

## R1 at the final header

`r1_newtable/`: the MSVC tree rebuilt at the header (VS 2026 18.11, 14.51, Release, 0 warnings),
the clang-cl tree rebuilt in round 5 (0 warnings); correctness `-LE "timing|benchmark"` 132/132 under
MSVC and 132/132 under clang-cl; `test_parallel_batch_gates` alone, opted out of throttling,
PARALLEL/VECTORIZED 0.235–0.254 at 1M for all eight operations (limit 0.75). An earlier R1 at the
round-2 header (fourteen drops only) read the same: 0 warnings, 132/132 each, gates 0.228–0.250.

## Files

- `round{1..5}/`: per tier (round 1) or AVX-512 only, `tab_{old,new}_{1,2}.csv`,
  `table_ab_analysis.txt`, `run.txt`, `environment.txt`.
- `rows_kAvx512.txt`, `summary.txt`: `derive_threshold_rows.py --tables kAvx512` at the final rules.
- `dispatch_thresholds.diff`: the header change against `122aa51` (21 values on 21 rows;
  clang-format 20.1.8).
- `gamma_probe/`: the probe and its output.
- `run_ab.ps1`: the ABAB runner (throttling opt-out, warm-up, quiet gate); `table_ab_analyze.py`:
  M's analyzer taking its directory as an argument; `ab_rows.py`: per-pass detail per (dist, op).

# A/B: current kNeon vs proposed kNeon, AUTO dispatch (M, 2026-10-14 18:4x, indicative)

Same code (d1adc95) in two Release trees: build-release (current header) and the detached worktree
../libstats-rows with dispatch_thresholds.h.proposed dropped in (0 warnings). tools/bench/v242_cost_bench.cpp
built against each; run back to back outside the sandbox on a busy machine (1-min load 7.9 -> 6.4), so
indicative only. CSVs: ab_oldtable.csv, ab_newtable.csv; v242_compare.py output: ab_compare.txt.

Noise floor: the 744 forced-strategy rows (same code, same table-independent paths) read new/old
median 0.99, p10 0.96, p90 1.03.

AUTO rows (216): new/old median 0.97. 47 rows faster by more than 1.25x, 0 slower by more than 1.25x.
- Discrete batches (Binomial, NegativeBinomial pdf/logpdf/cdf) at n = 1e4-1e5: 0.20-0.34x (3-5x faster).
- von Mises CDF at 1e4-1e5: 0.19-0.30x.
- Beta pdf/logpdf at 1e5: 0.24-0.30x; at 1e4: 0.54-0.63x.
Every one of those rows now runs AUTO within ~1.1x of its best forced strategy (was 3-5x off).

AUTO more than 1.25x slower than the best forced strategy: 53 rows under the current table, 10 under the
proposed one. The ten that remain, all with PARALLEL best:
- von Mises logpdf at 1e5, 2.8-3.0x (three shapes), and von Mises pdf at 1e5, 1.36x. The kNeon rows are
  standing overrides (PDF 100k, LogPDF 500k); the 2026-10-05 and 2026-10-14 captures both measure 25k,
  so the script kept them as KEEP override. Re-reading the row comment is the user's call: it argues
  from the pre-#111 memory-bound CDF, not the densities.
- Gamma(20) and Gamma(1000) pdf/logpdf at 1e4, 1.3-2.0x; Student-t(1000) pdf at 1e4, 1.42x. Shape
  dependence: the profile's sustained crossover for the distribution's default shape sits at 25k-50k,
  the bench's large shapes cross by 1e4. A per-distribution table cannot encode that; v3 input.
- NegativeBinomial(1000) logpdf at 1e5, 1.27x: at the edge of the 1.25 criterion.

The repo header is unchanged; the scratch worktree ../libstats-rows holds the proposed header.

## Pass 2 (18:52): von Mises PDF/LogPDF overrides dropped on kNeon (100k/500k -> 25k/25k)
Same method; CSVs ab2_oldtable.csv / ab2_newtable.csv; load 5.6 -> 3.9. Noise floor (forced rows) median 1.00,
p10 0.96, p90 1.04. AUTO: 49 rows faster > 1.25x, 1 slower (Student-t(7) cdf at 1e4: both tables dispatch
PARALLEL there, forced PARALLEL unchanged at 30 ns, so a scheduling outlier, not the table).
- von Mises logpdf at 1e5: 3.95 -> 1.40 ns/element (0.35x); AUTO/best 3.0 -> 0.9-1.1. The dropped override did it.
- von Mises pdf at 1e5 was already PARALLEL under the old 100k row (1e5 >= 100k); unchanged.
AUTO > 1.25x behind best forced: 54 rows under the current table, 7 under the proposed: Gamma(20), Gamma(1000),
Student-t(1000), Student-t(1e6) densities at 1e4 (1.3-1.95x; shape dependence, v3 input) and the Student-t(7)
cdf outlier. Verdict: better than pass 1, no table-caused regression.

## All-distribution bench (tools/bench/table_ab_bench.cpp, new): AUTO vs forced VECTORIZED/PARALLEL,
## 27 distributions x pdf/logpdf/cdf x 10 sizes (1024 .. 500k), ABAB passes, strategy_profile's parameters
Round 2 (18:53, header with the von Mises drop; round2/): noise floor p05-p95 0.93-1.13. AUTO rows 810: 157
faster > 1.25x, 11 slower. AUTO > 1.25x behind best forced: 195 (current table) -> 47. Real table-caused
regressions: Exponential CDF and Rayleigh CDF at 8192-10k (PARALLEL loses 1.6-2x to VECTORIZED for these
2.5 ns/element kernels; wins only from 25k) -> both kept at 25k (KEEP_ROW). The other "slower" rows are
unchanged table rows under load (per-pass data in table_ab_analysis.txt). Rows the rules protected but the
bench showed wrong under both tables, same gap and direction at every size: sub-fork table values
(Exponential/Laplace LogPDF 64, Uniform CDF 64, ChiSquared 512/128/64, Poisson CDF 256) -> new general rule;
Gaussian CDF NEVER (PARALLEL 1.25x at 10k .. 4.7x at 500k), Discrete CDF 1000k (1.7-3.7x from 100k),
Exponential PDF 25k (VECTORIZED best through 50k), Gumbel PDF 75k (1.45x at 50k), FisherF PDF 8192 (1.52x at
4096) -> DROP_OVERRIDE with the evidence. Also seen, not a table matter: Uniform CDF AUTO 3.3-4.6x behind
FORCE_VECTORIZED at 2k-10k under both tables with the old 64 row (PARALLEL forking on a trivial kernel).
Round 3 (18:59, header with those changes except FisherF PDF; tab_*_{1,2}.csv, table_ab_analysis_round3.txt):
noise floor 0.89-1.14 (load 6.9-8.7). AUTO: 191 faster > 1.25x, 29 slower; AUTO > 1.25x behind best: 202 -> 24.
The 29 slower are FisherF, Erlang, Gamma, von Mises, Student-t at 1e5-5e5 where both tables dispatch PARALLEL
and forced PARALLEL slowed by the same factor in the new passes: load, not the table. The 24 remaining gaps:
fork-zone rows at 2048 (Beta/Gamma/ChiSquared CDF 1.55-1.6x, FisherF PDF fixed in the final header), and
1.3-1.66x at one size for rows whose candidate sits one grid step below the table (Weibull LogPDF, Logistic
PDF/LogPDF, LogNormal PDF, Student-t PDF at 25k; Pareto PDF 50k): within the rules' "ok" band; optional.
Verdict: the proposed kNeon is applied to the worktree header for M's R1 (uncommitted).

## Round 4 (23:29, load 2.3-5.3): the eight TAKE_CANDIDATE rows, against the 580b7f7 header (round4/)
Gamma CDF, Beta CDF 4096 -> 2048; Weibull LogPDF, Logistic PDF/LogPDF, LogNormal PDF 50k -> 25k; Pareto PDF
75k -> 50k; Gaussian PDF 50k -> 75k. Noise floor (forced rows) p05-p95 0.91-1.08. All eight rows now run AUTO at
the best forced strategy's speed in both passes: Gamma/Beta/ChiSquared/Erlang CDF at 2048 from 34-42 ns to 21-31
(PARALLEL); Weibull/Logistic/LogNormal at 25k and Pareto at 50k from 3.2-5.5 ns to 2.0-3.5 (PARALLEL); Gaussian PDF at
50k from 1.9-2.3 to 1.5 (VECTORIZED), and TruncatedNormal LogPDF at 50k from 2.3 to 1.7-2.0 with it. AUTO > 1.25x behind
best forced: 13 -> 9; the remaining nine are two single-pass outliers on unchanged rows (Uniform at 2048 and von Mises
densities at 50k: first new pass 2-3x, second pass equal to the old, forced strategies unchanged) and five rows at
1.25-1.27x (Student-t, Erlang, ChiSquared PDF at 25k; Pareto CDF 50k), inside the noise band. R1 against the
header: 0 warnings, 132/132, gates 0.21-0.22. Committed from here.

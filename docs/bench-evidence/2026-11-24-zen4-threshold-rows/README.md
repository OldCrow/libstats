# kAvx512 threshold rows on the corvus adoption branch, 164cbce (2026-11-24)

The Zen 4 leg's L5 (`PLAN.md`, "Fleet legs for v3.0.0"). Input: the L4 bundle
`data/profiles/dispatcher/2026-11-24T07-53-40Z_windows-x86_64_dev-v3.0.0-corvus-adoption_sha-164cbce`
(three quiet runs with the sustained-v1 profiler; 3 of 81 rows spread more
than 2×).

**Nothing is applied.** `dispatch_thresholds.h` is unchanged in this tree.
The proposal below is the script's output; the A/B shows it is better than the
table on balance and wrong in a known way on about fifteen rows. **Decided
[user, 2026-11-24]: hold every value**, the one 2× row included; reconsider
if another machine's capture moves its table materially.

## Derivation

`scripts/derive_threshold_rows.py --tables kAvx512`, its kAvx512 entry
repointed (new: this bundle; previous: `2026-10-14T02-41-05Z`).
`rows_kAvx512.txt` has every row and decision, `derive_summary.txt` the
counts, `dispatch_thresholds.h.proposed.txt` the script's header,
`dispatch_thresholds.diff` its difference from the table.

Of 81 rows it keeps 42 and marks 39 `CHANGE`. Five of those restate the value
the table holds (the 2026-10-15 `BENCH_CROSSOVER` and `TAKE_CANDIDATE`
entries: Beta PDF and LogPDF, Cauchy CDF, Pareto CDF, TruncatedNormal LogPDF).
**34 values change:**

- **30 earlier.** 24 rows go from 20k–35k to 8192 or 10240; Erlang LogPDF
  75k → 10240, Gumbel, LogNormal and Logistic 50k → 10240 (LogPDF, LogPDF,
  CDF), Cauchy LogPDF 75k → 32768, InverseGamma LogPDF 200k → 75k. 14 of the
  30 are `CHANGE fork` (candidate at or below 8192, encoded as 8192).
- **4 later.** Bernoulli CDF and HalfNormal CDF 8192 → 10240, LogNormal CDF
  6144 → 8192 (the fork value), InverseGamma PDF 25k → 75k.

The three runs give the same crossover on most rows, so this is not noise. It
is the first derivation from a sustained-v1 capture (#320); the 2026-10-14
bundle the table came from used three warm-up calls and seven timed calls per
cell.

## A/B

`tools/bench/table_ab_bench.cpp` built with clang-cl and the library's own
flags against the tree (old table) and against a worktree at `164cbce` with
the proposed header (new, all 34 values). Run old, new, old, new from
`quiet.ps1` (phase `ab`; the script is in `../2026-11-24-zen4-v300-l4/`),
16 min a pass, every process opted out of power throttling. CPU 4.6% at the
gate; clock probe 127.3/127.1 ms before and 127.3/127.0 after (boost);
`noise.txt` never exceeds 0.16 of a core. `tab_{old,new}_{1,2}.csv` are the
passes, `ab_summary.txt` is
`../2026-10-15-zen4-threshold-rows/table_ab_analyze.py` on them.

- Noise floor (forced rows, same code): p05–p95 0.95–1.04.
- AUTO rows more than 1.25× behind their best forced strategy: **35 under the
  old table, 15 under the new.** 22 of 810 AUTO rows are faster by more than
  1.25×, 3 slower.
- The 19 gaps the new table closes are at 8192–25000 on rows that moved from
  25k: AUTO mostly 0.68–0.78× its old time there.

### What the new table gets wrong

**A notch at 10000.** On most rows PARALLEL beats VECTORIZED at 8192, loses
at 10000, and wins again from 25000. Medians of the four passes, ns/element:

| Row | P/V at 8192 | P/V at 10000 | P/V at 25000 |
|---|---|---|---|
| HalfNormal CDF | 0.77 | 0.95 | 0.61 |
| Bernoulli CDF | 0.73 | 1.14 | 0.77 |
| LogNormal LogPDF | 0.65 | 1.10 | 0.72 |
| LogNormal CDF | 0.69 | 0.73 | 0.45 |

The profiler's rule is "the smallest size from which PARALLEL wins at every
larger size", so it reads these rows as 10240, and the 15 rows the script
sets to 10240 leave AUTO on VECTORIZED at 8192, where PARALLEL is 1.3–1.6×
faster: 13 of the 15 remaining gaps are those rows at n = 8192
(`ab_summary.txt`). Setting them to 8192 instead would put AUTO on PARALLEL at
10000, about 1.1× behind VECTORIZED there. The old 25k values avoided the
notch by staying out of the zone. Why PARALLEL costs more per element at
10000 than at 8192 was not investigated; it looks like the slice arithmetic
`PLAN.md` describes under "Pool wake-up and batch spacing".

**The three regressions** (new AUTO more than 1.25× slower than old):

- HalfNormal CDF at 8192, 1.31×: the 8192 → 10240 change, the notch above.
  The old value was right at 8192.
- Cauchy LogPDF at 8192, 1.29×: AUTO moves to PARALLEL there although
  Cauchy's own LogPDF value is 32768. Cauchy's densities dispatch under
  Student-t's row, whose LogPDF goes 25k → 8192 in the proposal; PARALLEL is
  1.25× behind VECTORIZED for Cauchy at 8192 and 10000 while it is ahead for
  Student-t. One row serves two kernels with different crossovers.
- HalfNormal LogPDF at 1024, 1.32×: one pass of four (1.99 against 1.79 in
  the other), on an unchanged row below the fork value. Noise.

## Rows with a 2× win below their threshold

From the same four passes: rows whose table value sits above a measured size
where PARALLEL is already at least 2× ahead of VECTORIZED. There is one,
Erlang LogPDF (table 75k): P/V 0.68 at 8192, 1.14 at 10000, 0.75 at 25000,
0.46 at 50000, which supports 50k, not 8192. No row with a value above 8192
has a 2× win at 8192; the best is TruncatedNormal PDF at 0.63, and at 10000
those rows read 0.75–1.17. The rows where PARALLEL is 2× ahead at 8192 already
hold 8192 (35 of the 81 values are at or below it). The proposal's gains are
therefore all in the 1.3–1.6× band. The bench's case names did not match a
few table rows (NegBinomial among them), which this check skipped.

## Against L7

The profiler and this bench both call batches back to back. L7
(`../2026-11-24-zen4-batch-gap/`) shows that with 2 ms or more between
batches PARALLEL costs 1.9–2.6× more per element on this machine, and
Gaussian's crossover moves from 50000 to 100000. Every value this proposal
moves earlier is measured in the regime where the pool's workers never rest.
`PLAN.md` holds K's L5 on the pool design for the same reason, and leaves
open whether rows are re-derived after L7.

## L1 on the proposed header

`164cbce` with the proposed header, clang-cl, full build and
`ctest -LE "timing|benchmark" -j8`: 135/135 (`ctest_proposed_header.txt`).

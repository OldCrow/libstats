# kNeon threshold rows on the corvus adoption branch, cfbb992 (2026-11-22)

The M1 leg's L5 (PLAN.md "Fleet legs for v3.0.0"). Input: the L4 bundle
`data/profiles/dispatcher/2026-11-22T17-20-47Z_darwin-arm64_dev-v3.0.0-corvus-adoption_sha-cfbb992`
(three runs, `mediaanalysisd` alongside each; 34 rows spread more than 2x).

## Derivation

`scripts/derive_threshold_rows.py --tables kNeon`, its kNeon entry repointed
(new: this bundle; previous: 2026-10-14T04-39-37Z). `rows_kNeon.txt` has every
row and decision, `summary.txt` the counts, `dispatch_thresholds.h.proposed.txt`
the script's header. Of 81 rows it keeps 65 and proposes 11 changed values:

- 8 `CHANGE`, each the maximum of a > 2x spread that clears the table by more
  than one grid step, all later: Bernoulli CDF 6144 -> 25k, ChiSquared PDF
  25k -> 75k, Gamma LogPDF 25k -> 75k, InverseGamma LogPDF 25k -> 75k,
  NegBinomial LogPDF 2048 -> 6144, Poisson LogPDF 2048 -> 10k, Rayleigh PDF
  20k -> 75k, Rayleigh LogPDF 25k -> 75k.
- 3 `CHANGE taken` that move a value: Logistic LogPDF 25k -> 50k, Pareto PDF
  50k -> 75k, Weibull LogPDF 25k -> 10k. These come from the script's
  2026-10-14 `TAKE_CANDIDATE` entries, whose evidence names the October
  candidate; each is one grid step from the table, which rule 1 keeps. The
  other five `CHANGE taken` rows propose the value the table holds.

## A/B

`tools/bench/table_ab_bench.cpp` built against the tree (old table) and a
worktree at cfbb992 with the proposed header (new, all 11 values), `-O2`, run
old, new, old, new by `l5_ab_m_20261122.sh`, each pass gated on ambient CPU
below 15% (`run.txt`; passes started at 7.8%, 7.2%, 10.1%, 7.4%; 2 min each;
`mediaanalysisd` appeared in new pass 1, `noise_ab_*.txt`). `tab_*.csv` are
the raw passes, `ab_rows.txt` PARALLEL/VECTORIZED and AUTO/best per pass for
the 11 rows. Forced strategies run the same code under both tables, so each
size has four PARALLEL/VECTORIZED measurements.

| Row | Proposed | PARALLEL / VECTORIZED, four passes | Verdict |
|---|---|---|---|
| Poisson LogPDF | 2048 -> 10k | 1.38-1.49 at 2048, 1.29-1.82 at 4096, 1.07-1.23 at 8192, 1.04-1.14 at 10k, 0.58-0.63 at 25k | **applied**: AUTO at 2048-4096 goes from 1.35-1.47x behind the best forced strategy to level |
| Bernoulli CDF | 6144 -> 25k | 2.8-3.8 at 4096, 1.04-1.30 at 8192-10k, 0.65-0.70 at 25k | not applied: the row is a pass-through (dispatch runs under Binomial's CDF row, 2048), so the value is inert; AUTO was the same under both tables |
| ChiSquared PDF | 25k -> 75k | 0.83-0.90 at 25k, 0.70-0.82 at 50k | kept at 25k |
| Gamma LogPDF | 25k -> 75k | 0.99-1.05 at 25k, 0.74-0.86 at 50k | kept at 25k |
| InverseGamma LogPDF | 25k -> 75k | 1.00-1.10 at 25k, 0.95-1.09 at 50k, 0.83-0.88 at 100k | kept at 25k: level to 50k, nothing to gain |
| NegBinomial LogPDF | 2048 -> 6144 | 0.90-1.06 at 2048, 0.59-0.63 at 4096 | kept at 2048 |
| Rayleigh PDF | 20k -> 75k | 0.67-0.75 at 25k, 0.51-0.53 at 50k | kept at 20k; 75k put AUTO 1.3-1.9x behind |
| Rayleigh LogPDF | 25k -> 75k | 0.74-0.85 at 25k, 0.57-0.65 at 50k | kept at 25k; 75k put AUTO 1.2-1.7x behind |
| Logistic LogPDF | 25k -> 50k | 0.72-0.84 at 25k | kept at 25k |
| Pareto PDF | 50k -> 75k | 0.72-0.76 at 50k | kept at 50k |
| Weibull LogPDF | 25k -> 10k | 1.09-1.15 at 10k, 0.63-0.69 at 25k | kept at 25k |

Seven of the eight spread-maximum candidates lost the A/B: the late runs in
those rows are the loaded capture, not a moved crossover.

## Result

`dispatch_thresholds.diff`: kNeon Poisson LogPDF 2048 -> 10000 and the note
above the table. Rebuilt Release; `ctest -LE "timing|benchmark" -j6` 135/135.

## Open

- Bernoulli CDF under AUTO runs PARALLEL from 2048 (Binomial's row) and is
  2.8-3.8x behind VECTORIZED at 4096 and 1.04-1.30x behind at 8192-10k (1.2 ns
  to 3.8 ns per element). Fixing it means Bernoulli dispatching on its own
  row, a dispatch change, not a table value. Not acted on.
- The three stale `TAKE_CANDIDATE` entries are still in the script.
- **Provisional.** The bundle and `table_ab_bench` here are cfbb992's, before
  the per-cell steady-state timing of `8c55bfc` (#320), and both ran with the
  polling work-stealing pool alive, which is PARALLEL's best case. The applied
  row moves PARALLEL later, so it does not depend on that state; the kept rows
  may. Re-capture follows L7 and the pool design (PLAN.md).

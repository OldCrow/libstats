# Dispatch-threshold rows at d1adc95 — proposal (M, 2026-10-14)

Script: `scripts/derive_threshold_rows.py` (uncommitted; one script for every table, the table's
bundle and fork threshold as parameters). Run: `python3 scripts/derive_threshold_rows.py --out <dir>`.
Outputs here: `rows_<table>.txt` (all 81 rows per table with runs, candidate, previous candidate,
decision, proposed value, comment flag, Z direction, note), `summary.txt`, `dispatch_thresholds.h.proposed`
and `dispatch_thresholds.diff` (162 values on 85 rows; 170 after the eight-row follow-up below). Every existing comment is kept; each changed row's trailing
comment is prefixed with a marker `// 2026-10-14: <Op old→new, ...>.` and a five-line note above each table names
the bundle and the script, so the older prose is marked as predating the re-derivation.
The proposed header compiles (`-fsyntax-only`, AppleClang 21). Nothing is applied to the repo header.

## Inputs
| table   | bundle (final head)                       | previous capture (code-reason test) | fork value / zone | Z direction check |
|---------|-------------------------------------------|-------------------------------------|-------------------|-------------------|
| kAvx512 |     29 |          16 |      20 |            16 |             45 |
| kAvx2   |     15 |          26 |      33 |             7 |             41 |
| kAvx    |      8 |          23 |      46 |             4 |             31 |
| kNeon   |     29 |          16 |      33 |             1 |             45 |

kNeon's fork value is 2048 because M's `get_min_elements_for_parallel()` is 1536 (printed by every
stress reach line) and the `--large` grid has no 1536 point. kSse2: no table (decided); the SSE2
bundles are not read here.

## Rules, in order, per (distribution, operation)
1. KEEP ok — the K-method candidate (median of three, or the max when the runs spread > 2x) is within one
   grid step of the table.
2. KEEP judgment — rows in `KEEP_ROW`: kept at the table value against the rules by the user's judgment,
   evidence recorded (kNeon Exponential CDF and Rayleigh CDF at 25k: on the loaded machine PARALLEL at
   8192-10k loses 1.6-2x to VECTORIZED for these 2.5 ns/element kernels; it matches only at 25k).
3. KEEP override — the candidate is within one grid step of the previous capture's candidate, so the
   table's disagreement is a standing deliberate override. Not applied when the table value is below the
   fork value (PARALLEL does not fork there, so 64/128/256/512 were never measured crossovers: 4 rows on
   kAvx512, 7 on kAvx2, 4 on kNeon), nor to rows in `DROP_OVERRIDE` (user decisions with the evidence
   recorded: kNeon von Mises PDF/LogPDF, Gaussian CDF, Discrete CDF, Exponential PDF, Gumbel PDF, FisherF PDF).
4. SKIP spread — runs spread > 2x and the maximum is within a step of the table (none fired: such rows are
   already KEEP ok).
5. CHANGE fork — candidate at or below the fork zone: "parallel once it forks"; encode the fork value.
6. CHANGE — encode the candidate (the max when the runs spread > 2x).
Flags on CHANGE rows: `yes` in the cmt column (a per-row comment, kept verbatim after the marker);
"Z direction ... DIFFERS" on K's tables (2 rows on kAvx2, 5 on kAvx, where Z's capped tier reads no move);
"table already overrode the previous capture".

## Counts
| table   | CHANGE | CHANGE fork | KEEP ok | KEEP override | values written |
|(kNeon also has 2 KEEP judgment rows)||||||
|---------|-------:|------------:|--------:|--------------:|---------------:|
| kAvx512 |     29 |          16 |      20 |            16 |             45 |
| kAvx2   |     15 |          26 |      33 |             7 |             41 |
| kAvx    |      8 |          23 |      46 |             4 |             31 |
| kNeon   |     29 |          16 |      33 |             1 |             45 |

kAvx512 comes out at 45 rows against the "about 30" estimated from Z's assessment. The difference is mostly rows
flagged "table already overrode the previous capture" (23). The flag makes either reading a filter; the per-row
files carry the evidence. The sub-fork rule accounts for 4 of them.

## What the changes are
- Every table: the discrete rows (Bernoulli, Binomial, Geometric, NegBinomial) and the von Mises CDF
  leave NEVER for the fork value (or a measured 2k–6k on kNeon); the Student-t CDF likewise.
- Continuous CDFs (Exponential, Rayleigh, Weibull, Pareto, Cauchy, Laplace, TruncatedNormal,
  HalfNormal) move from 25k–2M to the fork value or 4k–8k.
- Densities: von Mises PDF/LogPDF, Gamma/InverseGamma LogPDF and (on kAvx512) most continuous
  PDF/LogPDF rows come down to 25k–150k; Uniform PDF/LogPDF/CDF leave NEVER on kAvx512 and kAvx2/kAvx.
- Beta PDF/LogPDF leave NEVER everywhere (fork value on K, 8192/10k on M, 25k on Z).
- Discrete PDF/LogPDF on kNeon go the other way, 50k -> 200k (parallel wins later than the table says).

## Follow-up: eight one-step rows (TAKE_CANDIDATE, user 2026-10-14)
Rule 1 kept rows within one grid step of the table. Where the quiet candidate and the loaded A/B agreed, the
candidate was taken (kNeon): Gamma CDF, Beta CDF 4096 → 2048; Weibull LogPDF, Logistic PDF/LogPDF, LogNormal PDF
50k → 25k; Pareto PDF 75k → 50k; Gaussian PDF 50k → 75k. Round 4 in AB_RESULTS.md confirms all eight.

## After adoption
One `dispatch_thresholds.h` commit; then R1 on every machine (the header is compiled into every build);
CI on the push. R8 is not repeated.

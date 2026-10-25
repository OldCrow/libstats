# 2026-09-30 Zen 4 comparatives, warmed — QUIET, single frequency regime. THE RECORD for this machine

Same machine, libraries, prefixes and runner as `../2026-09-29-zen4-indicative/`
(configuration and the two-compiler setup are in that README). The eight
bench binaries were rebuilt from `tools/bench/` with the in-process warm-up
(`LIBSTATS_BENCH_WARMUP_SECONDS=25`, `repass.ps1`), libraries untouched.
Gate passed at 4.93%; per-target noise 1.9–5.3% except one 11.35% sample
before `elem_v250_cap`, whose five columns all scale by the same 1.5× against
the unwarmed run, so the burst did not reach the numbers. Screen locked,
`InventorySvc` stopped.

**Regime check.** Rows that were boosted in the two earlier passes now read
1.5× slower and rows that were already sustained are unchanged: `corvus_scaling`
n = 1 gamma_p 1,403 ns (was 926–939) while n ≥ 2 rows are identical;
`dist_v241` gamma pdf-scalar 23.4 (was 15.7) while binomial 277 is unchanged;
`elem_*` every column ×1.5. So every row here is in the sustained regime,
which is the regime a batch workload runs in on this box and ~1.5×
pessimistic for a lone scalar call. The mid-n erf/exp rows of
`corvus_scaling` (n = 32–4096, 4.0–4.8 ns) are a noise blip — sub-100-ns
calls, and the n = 65536 rows agree with both earlier passes.

## corvus per element, n = 65536, ns

| | clang-cl `AVX3_ZEN4` | MSVC `AVX2` | M1 NEON |
|---|---:|---:|---:|
| erf | 2.4 | 12.2 | 2.2 |
| exp | 3.1 | 23.0 | 4.5 |
| lgamma | 28.7 | 134 | 45 |
| gamma_p | 90 | 2,722 | 190 |
| beta_p | 652 | 8,069 | 950 |
| gamma_p_inv | 1,094 | 10,699 | 2,400 |

Single call, n = 1, sustained: erf 30, exp 36, lgamma 246, gamma_p 1,403,
beta_p 2,772, gamma_p_inv 6,754 ns (MSVC AVX2: 75, 120, 611, 10,990, 19,806,
31,768). Per-call over per-element: gamma_p 15.6×, beta_p 4.3×,
gamma_p_inv 6.2× on AVX-512.

## Elementary family through `VectorOps`, n = 1e6, ns per element

| | v2.4.1 | branch, clang-cl corvus | branch, MSVC corvus |
|---|---:|---:|---:|
| exp | 1.03 | 3.10 (3.0× slower) | 23.1 (22×) |
| log | 1.37 | 7.37 (5.4×) | 57.2 (42×) |
| cos | 1.65 | 2.43 (1.5×) | 38.1 (23×) |
| sin | 1.65 | 2.43 (1.5×) | 41.3 (25×) |
| erf | 6.62 | **2.49 (2.7× faster)** | 12.3 (1.9× slower) |

## Distributions, ns per element, v2.4.1 → branch (clang-cl corvus) [→ MSVC corvus]

Batch at 1e6 under auto dispatch (12 threads); scalar at 1e5; quantile at 1e4.

| | pdf batch | logpdf batch | cdf batch | pdf scalar | cdf scalar | quantile |
|---|---|---|---|---|---|---|
| gamma | 0.8 → 2.3 (2.9×) [8.8] | 0.7 → 1.6 (2.3×) [6.4] | 12.1 → 11.7 (1.0×) [254] | 23.4 → 24.7 | 106 → 1,425 (13×) [11,326] | 800 → 7,752 (9.7×) [38,710] |
| beta | 6.9 → 21.4 (3.1×) [141] | 5.9 → 18.3 (3.1×) [118] | 8.7 → 75.7 (8.7×) [545] | 28.9 → 28.8 | 184 → 2,588 (14×) [17,125] | 1,032 → 18,052 (17×) [78,263] |
| poisson | 9.2 → 9.3 | 8.5 → 8.5 | 19.2 → 23.2 (1.2×) [432] | 99 → 99 | 180 → 1,678 (9.3×) [16,148] | 1,410 → 12,883 (9.1×) [135,212] |
| student_t | 2.9 → 11.9 (4.1×) [82] | 1.9 → 9.1 (4.8×) [58] | 206 → 623 (3.0×) [7,990] | 24.7 → 25.4 | 211 → 3,479 (16×) [25,588] | 1,060 → 61,629 (58×) [328,854] |
| binomial | 261 → 258 | 246 → 246 | 68 → 600 (8.8×) [2,618] | 277 → 277 | 567 → 4,200 (7.4×) [29,453] | 9,453 → 91,459 (9.7×) [610,021] |
| gaussian | 0.9 → 0.8 | 0.3 → 0.3 | 3.5 → 3.5 | 12.5 → 12.2 | 33.7 → 33.8 | 78 → 233 (3.0×) [340] |
| lognormal | 2.4 → 2.1 | 1.2 → 1.2 | 1.7 → 1.8 [7.4] | 21.3 → 21.3 | 32.5 → 71.2 (2.2×) [102] | 86 → 247 (2.9×) [355] |
| von_mises | 2.3 → 2.4 | 1.6 → 1.7 | 80 → 76 | 26.5 → 26.1 | 242 → 243 | 80 → 80 |
| weibull | 3.1 → 3.0 | 2.0 → 2.1 | 4.1 → 14.5 (3.5×) [105] | 32.5 → 31.8 | 30.9 → 30.8 | 37 → 37 |
| exponential | 0.9 → 0.9 | 0.4 → 0.4 | 0.9 → 0.9 | 12.3 → 12.0 | 12.2 → 11.8 | 13.6 → 14.2 |

Scalar pdf/logpdf: unchanged for every distribution (the PMF lgamma routing
was reverted before this branch state). Batch pdf/logpdf regressions are the
gamma family, beta and Student-t, through exp/log/lgamma. Batch CDF: gamma
and poisson at parity — `gamma_p` per element (90 ns) hides behind 12
threads; every regression above 3× is a `beta_p` consumer. Scalar CDF and
quantiles: the per-call price, 7–17× and 9–58×, as on the M1.

## Findings — what changed against the 2026-09-29 reading

Nothing in direction. The sub-2× rows are now resolved: batch pdf/logpdf
2.3–4.8× slower for gamma, beta and Student-t (the exp/log/lgamma cost);
poisson CDF batch 1.2×; lognormal scalar CDF 2.2×; weibull CDF batch 3.5×;
everything else at parity. The four provisional findings in
`../2026-09-29-zen4-indicative/README.md` stand with these numbers behind
them.

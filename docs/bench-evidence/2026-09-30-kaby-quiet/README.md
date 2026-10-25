# 2026-09-30 Kaby Lake comparatives — QUIET. THE RECORD for this machine

MacBook Pro 2017, i7-7820HQ (4 cores / 8 threads), macOS 13.7.8 Ventura,
AppleClang 15.0.0, CMake 4.4.3, Ninja. Both trees Release: worktree
`../libstats-v2.4.1` (tag `v2.4.1`, `692fd08`) built into its `build/`;
branch `dev/v2.5.0-corvus` at `6f112bb` built into `build-bench/`, corvus
v1.0.1 fetched via FetchContent against the system Highway 1.4.0 (Homebrew,
`/usr/local`). Both libraries warning-clean. libstats dispatches AVX2+FMA;
corvus reports `AVX2`. Benches compiled per `tools/bench/README.md`
(`scripts/build_bench.sh`), no warm-up (`LIBSTATS_BENCH_WARMUP_SECONDS`
unset — this machine has no frequency-regime step; see below). Runner:
corvus `tools/quiet_bench.sh -t AVX2 -m 5 -s 10`, unmodified. Gate passed at
4.10 / 3.27% after the desktop app went idle (the Claude helper and
WindowServer held ambient at 8–10% until then; `mediaanalysisd` and
`system_profiler` bursts to 18% in between); per-target noise before each
run 3.06–3.56%. One pass; no rerun owed.

**Regime check.** The corvus scaling rows are flat from n = 256 to 65536
(erf 6.5–6.7, gamma_p 148–152 ns) and `elem_v250` reads the same at 1e3 and
1e6, so there is no boosted/sustained split to reconcile on this box.
`corvus_scaling` n = 65536 erf 6.5 ns agrees with corvus's own Kaby Lake
record (7.38 at n = 1e6, `2026-08-30-kaby-quiet`).

## corvus per element, n = 65536, ns

| | Kaby Lake `AVX2` | Zen 4 clang-cl `AVX3_ZEN4` | M1 NEON |
|---|---:|---:|---:|
| erf | 6.5 | 2.4 | 2.2 |
| exp | 8.0 | 3.1 | 4.5 |
| lgamma | 58.3 | 28.7 | 45 |
| gamma_p | 152 | 90 | 190 |
| beta_p | 1,024 | 652 | 950 |
| gamma_p_inv | 1,886 | 1,094 | 2,400 |

Single call, n = 1: erf 35.9, exp 41.0, lgamma 274, gamma_p 1,258,
beta_p 2,727, gamma_p_inv 6,432 ns (Zen 4 sustained 1,403 / 2,772 / 6,754;
M1 827 / 1,500 / 4,300). Per-call over per-element: gamma_p 8.3×,
beta_p 2.7×, gamma_p_inv 3.4×.

The M1 session's expectation of "~2× better per element on AVX2" did not
hold: Kaby Lake AVX2 is at or behind M1 NEON per element on every column
(erf 3×, exp 1.8×, lgamma 1.3×, beta_p 1.08× slower; gamma_p and
gamma_p_inv 1.25× faster). Four AVX2 lanes do not make up for the 2017
core; corvus's own fleet evidence already had the Kaby-vs-M1 erf ratio at
3.35×. The single-call price is the same order on all three machines.

## Elementary family through `VectorOps`, ns per element

| | v2.4.1 @1e6 | branch @1e6 | v2.4.1 @1e3 | branch @1e3 |
|---|---:|---:|---:|---:|
| exp | 2.14 | 8.10 (3.8× slower) | 5.75 | 8.37 |
| log | 2.96 | 18.83 (6.4×) | 8.27 | 19.64 |
| cos | 2.61 | 4.29 (1.6×) | 6.91 | 4.41 |
| sin | 2.54 | 4.62 (1.8×) | 7.18 | 4.37 |
| erf | 9.01 | **6.70 (1.3× faster)** | 26.46 | 6.91 |

At 1e6 the direction matches Zen 4 (exp 3.0×, log 5.4×, cos/sin 1.5×
slower; erf 2.7× faster there): the x86 erf inversion holds on AVX2 with a
smaller margin, and exp/log are the loss. The v2.4.1 column is 2–3× slower
at 1e3 than at 1e6 while the branch is flat, so at 1e3 the branch wins on
cos, sin and erf and nearly ties on exp; that is a fixed per-call cost in the
retired v2.4.1 path, not a corvus property, and the 1e6 rows are the ones
that decide the elementary question.

## Distributions, ns per element, v2.4.1 → branch

Batch at 1e6 under auto dispatch (8 threads); scalar at 1e5; quantile at 1e4.

| | pdf batch | logpdf batch | cdf batch | pdf scalar | cdf scalar | quantile |
|---|---|---|---|---|---|---|
| gamma | 1.6 → 5.7 (3.6×) | 1.1 → 3.9 (3.5×) | 14.0 → 25.3 (1.8×) | 58.3 → 57.0 | 124 → 1,473 (12×) | 1,153 → 8,075 (7.0×) |
| beta | 13.8 → 56.9 (4.1×) | 11.3 → 47.3 (4.2×) | 13.5 → 157 (12×) | 68.6 → 68.3 | 146 → 2,889 (20×) | 748 → 19,598 (26×) |
| poisson | 7.5 → 6.8 | 5.3 → 5.5 | 17.8 → 48.9 (2.7×) | 75.3 → 73.9 | 105 → 1,748 (17×) | 758 → 13,685 (18×) |
| student_t | 3.9 → 4.1 | 2.4 → 2.7 | 25.6 → 213 (8.3×) | 59.4 → 62.8 | 188 → 3,803 (20×) | 851 → 64,233 (75×) |
| binomial | 64.9 → 64.9 | 52.8 → 53.9 | 233 → 4,548 (20×) | 119 → 117 | 237 → 4,491 (19×) | 3,841 → 96,595 (25×) |
| gaussian | 1.4 → 1.4 | 0.6 → 0.6 | 5.8 → 6.0 | 51.6 → 52.7 | 70.4 → 69.2 | 165 → 259 (1.6×) |
| lognormal | 3.6 → 3.7 | 1.7 → 1.7 | 3.2 → 5.3 (1.7×) | 57.3 → 57.1 | 79.2 → 95.8 (1.2×) | 161 → 277 (1.7×) |
| von_mises | 4.8 → 4.1 | 2.7 → 2.7 | 143 → 199 (1.4×) | 64.7 → 63.9 | 385 → 372 | 143 → 137 |
| weibull | 5.6 → 5.8 | 3.4 → 3.5 | 5.7 → 6.6 (1.2×) | 75.6 → 74.9 | 74.7 → 73.8 | 74.1 → 72.5 |
| exponential | 1.6 → 1.5 | 0.6 → 0.7 | 1.5 → 1.5 | 50.8 → 50.5 | 50.2 → 50.7 | 51.3 → 49.9 |

Scalar logpdf (not shown): parity for every distribution, as scalar pdf.

## Findings

- **Scalar pdf/logpdf at parity everywhere; batch pdf/logpdf regress only
  for gamma (3.6×) and beta (4.1×)**, through exp/log. Student-t batch
  pdf is at parity here (Zen 4: 4.1–4.8× slower) — its v2.4.1 path on
  AVX2 was already paying for a log it now gets from corvus at similar cost.
- **Batch CDF regresses more than on Zen 4 and less than on the M1**:
  gamma 1.8× (Zen 4 parity, M1 3×), poisson 2.7× (1.2×, 4×), beta 12×
  (8.7×, 23×), student_t 8.3× (3.0×, 12×), binomial 20× (8.8×, 18×). Eight
  threads hide less of `gamma_p`'s 152 ns than twelve hide of 90; the
  `beta_p` consumers dominate at 12–20×. Weibull batch CDF 1.2× (Zen 4
  3.5×).
- **Scalar CDF 12–20× and quantiles 7–75×** — the per-call price, same
  shape as both other machines; Student-t quantile 75× (Zen 4 58×) is the
  worst row on the fleet. Gaussian/lognormal quantile 1.6–1.7× (Zen 4 3×,
  M1 1.4×) via the corvus probit.
- **Elementary, AVX2**: exp 3.8× and log 6.4× slower, cos/sin 1.6–1.8×
  slower, erf 1.3× faster at 1e6. Same verdict as Zen 4: keep corvus erf
  on x86, exp/log (and cos/sin) are the loss increment 4 has to answer for
  — revert them on x86 or corvus #43-style kernels for x86 too.
- **corvus fleet targets (per element, AVX2)**: erf 6.5, exp 8.0, lgamma
  58, gamma_p 152, beta_p 1,024, gamma_p_inv 1,886 ns; single call gamma_p
  1.26, beta_p 2.7, gamma_p_inv 6.4 µs. AVX2 sits between Zen 4 and the
  M1 on the incomplete-gamma family and behind the M1 on erf/exp/lgamma.

# 2026-09-29 Zen 4 comparatives, v2.4.1 → dev/v2.5.0-corvus — INDICATIVE, not a quiet pass

Asus TUF A16, Ryzen 7 7445 (Zen 4, AVX-512), Windows 11 Pro, on AC, Turbo
power scheme. libstats MSVC 19.51.36260 `/arch:AVX512`, Ninja Release, on
both sides: worktree at tag `v2.4.1` (OLD) and the branch at `6384da6` (NEW).
Benchmarks: `tools/bench/` at `6384da6`, compiled with the library's own
flags (`scripts/build_bench.ps1`). Runner: corvus `tools/quiet_bench.ps1`,
unmodified, 5% gate.

Two NEW configurations, both corvus v1.0.1 + Highway 1.4.0 installed as a
system package (`scripts/build_deps.ps1`):

| Suffix | corvus + Highway compiler | corvus dispatch |
|---|---|---|
| (none) | clang-cl 22.1.3 (VS-bundled) | `AVX3_ZEN4` |
| `_cap` | MSVC 19.51 | `AVX2` — Highway's MSVC blocklist; what a default FetchContent build on Windows produces |

## Why INDICATIVE

- **Noise.** The gate passed at 2.84% and both `corvus_scaling` runs held
  2.7–3.2%, but the `elem_*` and `dist_*` runs saw 4–12.5% (`runner_log.txt`).
  The runner does not record consumers after the gate, so the source is
  unknown.
- **Frequency regime.** Single-thread throughput steps down ~1.5× about 15 s
  into sustained load and stays there — the Zen 4 frequency-scaling artifact
  (`docs/VALIDATION_HISTORY.md`). Reproduced after the pass with a second
  `corvus_scaling` run (identical rows). Evidence inside the files: the
  `n = 1` row of `corvus_scaling` matches `smoke_run.txt` and every later
  row is 1.5× slower; in `dist_v241` the gamma–student_t rows match the
  smoke run and binomial onward are 1.5× slower; in `dist_v250` only the
  gamma row is boosted. The step lands on different rows of OLD and NEW, so
  **ratios under ~2× in the distribution tables are not reliable**. Ratios
  from rows in one regime are, and `smoke_run.txt` (all rows boosted, but
  under 31–56% load) cross-checks them.
- Neither `elem_*` run shows the step (4 s each); the elementary numbers
  agree with the smoke run to within 15%.

## Results

corvus called directly, per element at n = 65536, sustained regime, ns.
M1 column from `PLAN.md` (NEON, 2026-09-29, M1 session):

| | clang-cl `AVX3_ZEN4` | MSVC `AVX2` | M1 NEON |
|---|---:|---:|---:|
| erf | 2.4 | 12.2 | 2.2 |
| exp | 3.1 | 23.0 | 4.5 |
| lgamma | 29 | 135 | 45 |
| gamma_p | 90 | 2,726 | 190 |
| beta_p | 649 | 8,056 | 950 |
| gamma_p_inv | 1,093 | 10,699 | 2,400 |

Single call (n = 1, boosted): gamma_p 926 ns, beta_p 1,815 ns — the M1's
827 and 1,500. The per-call cost is tier-independent, as predicted.

Elementary family through `VectorOps` at n = 1e6, ns per element:

| | v2.4.1 | branch, clang-cl corvus | branch, MSVC corvus |
|---|---:|---:|---:|
| exp | 0.68 | 2.09 (3.1× slower) | 15.4 (23×) |
| log | 0.83 | 4.92 (5.9×) | 38.2 (46×) |
| cos | 1.10 | 1.62 (1.5×) | 25.5 (23×) |
| sin | 1.10 | 1.62 (1.5×) | 27.5 (25×) |
| erf | 4.42 | **1.65 (2.7× faster)** | 8.25 (1.9× slower) |

Distributions, v2.4.1 → branch, same-regime rows only:

| | clang-cl corvus | MSVC corvus | M1 |
|---|---|---|---|
| scalar CDF (gamma, beta, poisson, student_t, binomial) | 7–15× slower | 50–180× | 12–21× |
| quantile, same five | 10–57× slower | 60–470× | 8–66× |
| batch CDF at 1e6, auto dispatch: gamma, poisson | parity | 18–24× | 3–4× |
| batch CDF at 1e6: beta / binomial / student_t | 8× / 8× / 3–5× | 69× / 36× / 59× | 23× / 18× / 12× |
| gaussian, lognormal quantile | 3× | 4× | 1.4× |
| gaussian, weibull, exponential, von Mises otherwise | ≤ 1.5× | weibull cdf 26× via exp/log | ≤ 1.5× |

## Provisional findings

1. **The MSVC build of corvus is 5–30× slower than the clang-cl build.**
   corvus measured the AVX2-vs-AVX-512 width difference at ~1.6× on this CPU
   (corvus `CMakeLists.txt`, 2026-07-24), so most of the gap is MSVC code
   generation of Highway code, not the tier. Unseparated here: a clang-cl
   build capped at AVX2 (`CORVUS_DISABLED_TARGETS`) would isolate it. Until
   then, any Windows consumer of a FetchContent branch build pays this.
2. **erf inverts between ISAs.** corvus erf is 2.7× faster than the retired
   musl-polynomial x86 kernel at AVX-512 and 1.6× slower than the retired
   table kernel on NEON. exp and log lose on both ISAs (x86 3–6×, NEON
   4.6–8×); cos/sin lose ~1.5× on both. So the increment-4 question has a
   per-ISA answer for erf only.
3. **Batch CDF is better on x86 than on the M1** — gamma and poisson reach
   parity under auto dispatch (8 lanes plus 12 threads) — but scalar CDF and
   quantiles are as slow as on the M1. The single-call path (corvus #42's
   scalar entry point) is the fleet-wide cost; the batch kernels are the
   NEON-specific one.
4. **Gaussian and lognormal quantiles are 3× slower here vs 1.4× on the M1**
   (corvus erfinv per single call).

## Owed

A re-pass holding one frequency regime across both binaries (run each twice
back to back and keep the second), under the gate for the whole pass. The
binaries and prefixes are on the machine under `build-bench-msvc*/` and
`build-bench-deps/`; `scripts/` here is the exact recipe that built them.

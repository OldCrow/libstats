# Z at 164cbce: v3.0.0 (corvus v1.2.0) fleet legs L1–L4, 2026-11-23/24

Zen 4's legs for `dev/v3.0.0-corvus-adoption` (`PLAN.md`, "Fleet legs for
v3.0.0") at `164cbce` (last code change `e4026b3`). L7 is in
`../2026-11-24-zen4-batch-gap/`, L5 in `../2026-11-24-zen4-threshold-rows/`,
the dispatcher bundle in
`data/profiles/dispatcher/2026-11-24T07-53-40Z_windows-x86_64_…`.

Asus TUF A16, Ryzen 7 7445HS (6 cores / 12 threads, AVX-512), Windows 11 Pro
10.0.26300, Visual Studio 18 Community, CMake 4.4.4, Ninja 1.13.2, power
scheme Performance, on AC. Work was done in a separate worktree of the branch
with its own build directories.

## L1 — native correctness (`l1/`)

Fresh Release configures (`build.ps1.txt`): clang-cl 22.1.3 with Ninja (the
`windows-clang-cl` preset's settings) and MSVC 19.52.37029 (toolset 14.52)
with the Visual Studio generator. corvus v1.2.0 and Highway 1.4.0 fetched.

| | clang-cl 22.1.3 | MSVC 19.52 |
|---|---|---|
| Tier (`system_inspector --quick`) | AVX-512, 12 logical / 6 physical | the same |
| Correctness (`-LE "timing\|benchmark" -j8`) | 135/135 | 135/135 |
| `test_concurrency_stress` | passed, 206 s | passed, 195 s |
| `test_parallel_backend_selection` | 3 pass, the OpenMP case skips | the same |
| Build warnings | 2 | 8 |

- **Warnings were not zero.** clang-cl: two `-Wdeprecated-declarations` for
  `getenv` in `tools/bench/sustained_timing.h`, compiled into
  `tools/strategy_profile.cpp` since `8c55bfc`; the hand-built
  `distributions_bench` and `elementary_bench` warned the same way. Fixed in
  this change: the header gains `bench_timing::envValue()` (`_dupenv_s` on
  Windows, the pattern `tests/test_concurrency_stress.cpp` already uses) and
  the three callers use it. A full clang-cl rebuild and the four benches are
  then warning-free. `corvus_scaling_bench.cpp` and `v242_cost_bench.cpp`
  still call `std::getenv`; neither was built here.
  MSVC: eight `D9025` (`/EHs` and `/EHc` overridden) from Highway's own four
  projects, whose CMake adds `/EHs-c-` after CMake's default `/EHsc`. Third
  party, fetched through corvus; not changed. None of the ten was in `src/`
  or `include/`.
- **`test_concurrency_stress` passes natively under `cl.exe`**, which CI's
  `cl.exe` leg excludes. MSVC builds corvus capped at AVX2 (the configure log
  says so); the tier above is libstats's own.
- Backend: `Issue179_BackendMatchesTheBuild` passes under both, and its
  expectation with the MSVC standard library is `std::execution`.

## L2 — accuracy sweep and block (`accuracy/`)

Sweeps started from the clean worktree; banner
`commit=164cbce  isa=AVX-512  date=2026-11-23`. Python 3.14.10, mpmath 1.4.1.

- clang-cl (the sweep of record): 26,882 rows, 200 self-checks, **0 contract
  violations**, 0 fallbacks. The `isa=AVX-512` block of
  `docs/ACCURACY_CHARACTERIZATION.md` is regenerated (110 lines changed).
- MSVC, scored against a copy of the document: 0 contract violations.

## L2b — row gate and off-grid sweep (`accuracy/`)

- **Allowlist:** `data/accuracy/allow_AVX-512.csv`, 3,025 rows over budget
  (AVX 3,039, AVX2 3,037, NEON 2,971).
- **Row gate against v2.5.0 on the same inputs** (`oracle_rowgate_v250.txt`):
  `v2.5.0` built with clang-cl beside the checkout and swept; this branch
  swept with `--grid-from` that CSV and scored with `--baseline` the v2.5.0
  rows. 26,301 rows scored, **707 values better, 74 worse**, 0 contract
  violations. The 74, by group:

  | Group | Values | Worst (v2.5.0) |
  |---|---|---|
  | NegativeBinomial pdf and logpdf, r = 20 | 30 | 41 ε (10) |
  | Student-t pdf and logpdf, ν 30.9, 31, 39.9 | 18 | 33 ε (1.4) |
  | Student-t pdf, ν 1000, 1e7, 1e18 | 12 | 13 ε (3.6) |
  | Student-t cdf and survival, ν 39.9 | 3 | 13 ε (2.7) |
  | Fisher-F logpdf (2, 0.2) and (5, 10) | 4 | 311 ε (52) |
  | Beta pdf (0.01, 0.01) and (10, 0.1) | 3 | 6.4 ε (1.7) |
  | Geometric pdf, p = 0.3 | 2 | 16 ε (0.6) |
  | Fisher-F survival (1e4, 1e4) | 1 | 10 ε (2.2) |
  | LogNormal quantile (0, 2) | 1 | 7.0 ε (1.6) |

  These are the classes K and M reported: small-shape densities (#314) and
  rounding changes on ill-conditioned formulas (Student-t at ν 39.9 and at
  large ν, Fisher-F survival, LogNormal quantile, Geometric pdf). The same
  rows are over budget at similar scores in `allow_AVX.csv`, `allow_AVX2.csv`
  or `allow_NEON.csv`. No reference was re-checked at higher precision here.
- **Off-grid, seed 213** (`oracle_offgrid.txt`): 11,389 rows, 11,278 scored,
  60 skipped (oracle limits; the other machines skipped 141–142 at the same
  20-s row limit), **1 contract violation**: Fisher-F logpdf at
  x = `denorm_min`, shapes (46.9, 735048), −inf against a finite reference.
  That is #317. #316 no longer appears (fixed on `main`, merged here).
  Densities in the hundreds of ε at large shape or in a steep tail are the
  conditioning class the document describes.
- **Off-grid quantile rows over 100 ε: real, 14 rows**
  (`offgrid_quantile_check.py`, `offgrid_quantile_check.txt`). Each
  reference was recomputed at 130 digits by a route independent of the
  oracle (bisection on a CDF built from `mp.betainc` and `mp.gammainc`, on
  the smaller tail, with the exact double p; Weibull by its closed form) and
  reproduces the oracle's score on all 14, so none is an oracle error. The
  library's value is 117–620 ulps from the correctly rounded quantile:
  Student-t ×4 (ν 0.48–2.05, p below 1e-117; worst 474 ε), Fisher-F ×7 (one
  degree of freedom 0.07–0.12, p below 1e-56 or within 1e-9 of 1; worst
  398 ε), InverseGamma ×2 (shape below 0.18, p within 1e-11 of 1; 211 ε),
  Weibull ×1 (shape 0.094, p 6.8e-26; 115 ε). The common factor is a small
  shape with a deep tail; x runs from 1e-271 to 1e+247. **Not the adoption's:**
  the same 14 calls against `main` (`7069adc`, clang-cl;
  `offgrid_quantile_main_vs_dev.cpp` and `.txt`) return the same bits on 10
  rows (all four Student-t, Weibull, four Fisher-F, one InverseGamma). On the
  other four `main` is further from the reference than this branch: Fisher-F
  310, 707 and 329 ε against 203, 317 and 184 here, InverseGamma 302 ε
  against 211. This is #104's subject (the quantile contract at
  extreme p). Not filed.

## L3 — prototype bit-identity

`b766e45` and `90d1bef` built with clang-cl (the `accuracy_sweep` target) and
swept: the two CSVs are **identical past the banner lines**, 26,882 rows.

## L4 — quiet session

One detached script, `quiet.ps1.txt`; `run.txt` is its log. Every measured
process was launched directly and opted out of Windows power throttling.
Total CPU at each gate 2.1–3.3%; the 10-s clock probe read 126.9–127.4 ms per
pass before every phase and after the last (boost is 127–130, base clock
about 190). The sitting ran 02:42–06:58 local, after L7.

- **Timing tests** (`timing/`): 14/14, 19 test cases, each binary launched
  with ctest's own `GTEST_FILTER`.
- **`simd_verification`** (`simd/`): 140 pass, 0 fail.
- **Benches** (`bench/`): `distributions_bench.cpp` and `elementary_bench.cpp`
  from this checkout, built with clang-cl and the library's own flags
  (`build_bench.ps1.txt`) against `main` at `3dd8358` (code `7069adc`) and
  against this branch. Three passes with `LIBSTATS_BENCH_WARMUP_SECONDS=25`,
  main first on passes 1 and 3, dev first on pass 2. Sustained timer (#320).
  This is Z's number for #156.
- **Profile**: three `strategy_profile --large` runs, 81 min each; see the
  bundle's `manifest.txt`.

Median of three passes, main → dev (dev/main). Columns: batch at 1e6 under
AUTO, scalar at 1e5, quantile at 1e4.

| ns/elem | pdf@1e6 | lpdf@1e6 | cdf@1e6 | pdf-sc | lpdf-sc | cdf-sc | quantile |
|---|---|---|---|---|---|---|---|
| gamma | 0.5 → 0.5 (1.00×) | 0.3 → 0.3 (1.00×) | 24.0 → 20.4 (0.85×) | 16.5 → 16.6 (1.01×) | 11.2 → 11.3 (1.01×) | 108.2 → 411.1 (3.80×) | 541.7 → 5534.0 (10.22×) |
| beta | 0.8 → 0.8 (1.00×) | 0.7 → 0.7 (1.00×) | 10.6 → 60.5 (5.71×) | 20.4 → 20.5 (1.00×) | 14.3 → 14.4 (1.01×) | 117.9 → 1612.2 (13.67×) | 677.2 → 10594.0 (15.64×) |
| poisson | 9.1 → 4.8 (0.53×) | 8.6 → 4.1 (0.48×) | 18.3 → 37.0 (2.02×) | 45.1 → 24.8 (0.55×) | 38.9 → 19.6 (0.50×) | 80.5 → 581.8 (7.23×) | 322.3 → 1773.6 (5.50×) |
| student_t | 0.7 → 0.7 (1.00×) | 0.5 → 0.5 (1.00×) | 20.8 → 20.8 (1.00×) | 16.5 → 16.6 (1.01×) | 10.7 → 10.8 (1.01×) | 163.1 → 878.5 (5.39×) | 826.1 → 2324.8 (2.81×) |
| binomial | 18.0 → 10.9 (0.61×) | 15.9 → 8.9 (0.56×) | 28.4 → 86.2 (3.04×) | 83.3 → 52.2 (0.63×) | 73.9 → 43.6 (0.59×) | 133.9 → 1776.9 (13.27×) | 483.5 → 5122.9 (10.60×) |
| gaussian | 0.3 → 0.3 (1.00×) | 0.1 → 0.1 (1.00×) | 3.4 → 3.0 (0.88×) | 8.3 → 8.5 (1.02×) | 6.2 → 6.2 (1.00×) | 22.5 → 24.7 (1.10×) | 72.0 → 106.0 (1.47×) |
| lognormal | 0.5 → 0.5 (1.00×) | 0.3 → 0.3 (1.00×) | 1.4 → 1.0 (0.71×) | 14.3 → 14.3 (1.00×) | 9.3 → 9.3 (1.00×) | 21.5 → 19.8 (0.92×) | 76.1 → 116.5 (1.53×) |
| von_mises | 0.6 → 0.6 (1.00×) | 0.4 → 0.4 (1.00×) | 29.6 → 29.3 (0.99×) | 15.6 → 15.6 (1.00×) | 11.8 → 11.9 (1.01×) | 131.3 → 131.1 (1.00×) | 377.8 → 407.3 (1.08×) |
| weibull | 1.0 → 1.0 (1.00×) | 0.6 → 0.6 (1.00×) | 1.0 → 1.0 (1.00×) | 21.9 → 21.9 (1.00×) | 15.6 → 15.6 (1.00×) | 26.2 → 26.2 (1.00×) | 26.8 → 26.8 (1.00×) |
| exponential | 0.3 → 0.3 (1.00×) | 0.1 → 0.1 (1.00×) | 0.6 → 0.6 (1.00×) | 8.7 → 8.5 (0.98×) | 6.0 → 6.0 (1.00×) | 10.0 → 10.0 (1.00×) | 11.0 → 11.0 (1.00×) |

`elementary_bench` (retired tier kernels against corvus):

| ns/elem | exp | log | cos | sin | erf |
|---|---|---|---|---|---|
| n=1000 | 0.69 → 0.69 (1.00×) | 0.80 → 0.80 (1.00×) | 1.01 → 1.01 (1.00×) | 1.05 → 1.05 (1.00×) | 3.50 → 1.66 (0.47×) |
| n=1000000 | 0.69 → 0.69 (1.00×) | 0.80 → 0.80 (1.00×) | 1.02 → 1.03 (1.01×) | 1.05 → 1.05 (1.00×) | 3.48 → 1.66 (0.48×) |

The largest max/min over the three passes is 1.17 for a `distributions_bench`
cell and 1.01 for an `elementary_bench` cell (`bench/compare.txt`, from
`bench/bench_compare.py`).

Reading:
- Continuous densities are level. Poisson and Binomial pmf and log-pmf are
  0.48–0.63×, batch and scalar (the integer `lgamma` table, L3's prototype).
  `erf` is 0.47×; `exp`, `log`, `cos` and `sin` are level.
- The incomplete gamma and beta paths cost more, as #156 records for M and K.
  Batch CDF: Poisson 2.0×, Binomial 3.0×, Beta 5.7×; Gamma is faster here,
  0.85× (K 1.68×). Scalar CDF: Gamma 3.8×, Student-t 5.4×, Poisson 7.2×,
  Binomial 13.3×, Beta 13.7×. Quantile: Student-t 2.8×, Poisson 5.5×, Gamma
  10.2×, Binomial 10.6×, Beta 15.6×.
- **Gaussian and LogNormal quantile are 1.47× and 1.53× slower** (72 → 106
  and 76 → 117 ns). K measured both faster on this branch (0.82× and 0.84×).
  The code path explains both. On `main` the probit is libstats' own scalar
  code. Here `inverse_normal_cdf` is `corvus::erfcinv` called on a span of
  one element (`corvus_scalar` in `src/math_utils.cpp`): corvus has no scalar
  entry, so each call runs the vector kernel at full width for one value. In
  absolute terms the branch is faster on Z than on K (106 against 178 ns);
  the ratio differs because `main`'s scalar probit is 3.0× faster on Z than
  on K (72 against 216 ns, in line with the other scalar columns) while the
  one-element kernel call is only 1.7× faster. Not a defect of this tier. It
  is the cost corvus #110 describes for `lgamma` and `lbeta`, on `erfcinv`
  and `erfinv`.
- Gaussian and LogNormal batch CDF are 0.88× and 0.71×; Student-t batch CDF
  is level while its scalar CDF is 5.4×.

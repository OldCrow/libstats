# K at 255e1fe: v3.0.0 (corvus v1.2.0) against main, 2026-11-22

Kaby Lake i7-7820HQ (AVX2+FMA), AppleClang 15, macOS 13.7.8, screen locked, AC power.
`dev/v3.0.0-corvus-adoption` at `255e1fe` (last code change `ec8cdc9`) against `main` at
`99f2175` (v2.5.0 line), both Release. `tools/bench/distributions_bench.cpp` and
`elementary_bench.cpp` from the dev checkout, built against each tree's `libstats.a` with the
library's own flags (`build_bench.sh.txt`). Three passes, main then dev alternately, each
binary started once the 1-minute load was below 2.0 (`run.txt`; `environment.txt` has the
binary hashes). Fleet-leg step L4; this is K's number for #156. The same sitting's
dispatcher bundle: `data/profiles/dispatcher/2026-11-22T00-28-35Z_darwin-x86_64_…`.

Median of three passes, main → dev (dev/main). The largest max/min over the three passes of
any cell is 1.22. Columns: batch at 1e6 under AUTO, scalar at 1e5, quantile at 1e4.

| ns/elem | pdf@1e6 | lpdf@1e6 | cdf@1e6 | pdf-sc | lpdf-sc | cdf-sc | quantile |
|---|---|---|---|---|---|---|---|
| gamma | 1.6 → 1.6 (1.00×) | 1.1 → 1.1 (1.00×) | 23.4 → 39.3 (1.68×) | 58.0 → 59.2 (1.02×) | 52.1 → 53.7 (1.03×) | 170.4 → 719.9 (4.22×) | 838.9 → 8832.3 (10.53×) |
| beta | 2.3 → 2.3 (1.00×) | 1.8 → 1.8 (1.00×) | 16.5 → 134.5 (8.15×) | 72.8 → 71.0 (0.98×) | 58.7 → 57.6 (0.98×) | 160.4 → 2796.7 (17.44×) | 1053.8 → 17406.4 (16.52×) |
| poisson | 9.2 → 7.3 (0.79×) | 7.8 → 5.3 (0.68×) | 22.3 → 78.7 (3.53×) | 90.1 → 77.0 (0.85×) | 72.3 → 62.8 (0.87×) | 149.3 → 1035.2 (6.93×) | 528.6 → 3009.9 (5.69×) |
| student_t | 1.8 → 1.8 (1.00×) | 1.2 → 1.2 (1.00×) | 28.4 → 29.2 (1.03×) | 61.4 → 60.7 (0.99×) | 52.5 → 52.3 (1.00×) | 237.5 → 1521.1 (6.40×) | 1363.9 → 3938.3 (2.89×) |
| binomial | 23.3 → 20.6 (0.88×) | 20.7 → 17.8 (0.86×) | 40.5 → 192.3 (4.75×) | 169.3 → 150.2 (0.89×) | 148.3 → 131.7 (0.89×) | 243.9 → 2989.6 (12.26×) | 812.4 → 8340.2 (10.27×) |
| gaussian | 0.8 → 0.7 (0.87×) | 0.6 → 0.6 (1.00×) | 4.8 → 4.2 (0.88×) | 50.8 → 50.7 (1.00×) | 47.0 → 46.7 (0.99×) | 69.2 → 70.9 (1.02×) | 216.3 → 178.0 (0.82×) |
| lognormal | 1.5 → 1.5 (1.00×) | 1.0 → 1.1 (1.10×) | 3.2 → 2.6 (0.81×) | 60.5 → 60.1 (0.99×) | 55.2 → 54.5 (0.99×) | 84.4 → 83.0 (0.98×) | 233.6 → 196.9 (0.84×) |
| von_mises | 1.7 → 1.6 (0.94×) | 1.0 → 1.0 (1.00×) | 53.3 → 53.8 (1.01×) | 65.8 → 66.4 (1.01×) | 52.9 → 53.3 (1.01×) | 300.0 → 305.1 (1.02×) | 839.9 → 832.9 (0.99×) |
| weibull | 2.0 → 2.0 (1.00×) | 1.5 → 1.5 (1.00×) | 3.4 → 3.4 (1.00×) | 75.0 → 75.6 (1.01×) | 56.5 → 58.6 (1.04×) | 79.0 → 78.8 (1.00×) | 76.0 → 76.1 (1.00×) |
| exponential | 0.9 → 0.8 (0.89×) | 0.6 → 0.6 (1.00×) | 1.2 → 1.2 (1.00×) | 50.0 → 50.4 (1.01×) | 46.2 → 46.3 (1.00×) | 52.0 → 51.8 (1.00×) | 51.1 → 51.1 (1.00×) |

`elementary_bench` (retired tier kernels against corvus):

| ns/elem | exp | log | cos | sin | erf |
|---|---|---|---|---|---|
| n=1000 | 5.72 → 5.73 (1.00×) | 8.26 → 8.29 (1.00×) | 6.86 → 6.94 (1.01×) | 7.13 → 7.16 (1.00×) | 26.34 → 19.28 (0.73×) |
| n=1000000 | 2.23 → 2.29 (1.03×) | 2.95 → 2.90 (0.98×) | 2.54 → 2.46 (0.97×) | 2.54 → 2.50 (0.98×) | 8.98 → 6.49 (0.72×) |

Timing method: the benches at `255e1fe`, before `8c55bfc` (#320): a few calls per cell, no
per-cell warm-up, and `main` always ran before dev. On the Ivy Bridge machine that method
produced ratios of 0.72× and 1.44× that were the CPU clock recovering, not the code
(AGENTS.md, Testing). Whether macOS on this CPU has the same effect is not measured, so the
ratios within about 1.5× of level here are unconfirmed until re-run with the sustained
timer; the 3× to 17× ratios are outside what that effect explains.

Reading:
- Densities are level or faster. Poisson and Binomial pmf/log-pmf are 0.68–0.89× (the
  integer `lgamma` table, L3's prototype). Gaussian and LogNormal CDF and quantile are
  0.81–0.88×; `erf` is 0.72×.
- The incomplete gamma and beta paths cost more, as #156 records for M. Batch CDF: Gamma
  1.7×, Poisson 3.5×, Binomial 4.8×, Beta 8.2×. Scalar CDF: Gamma 4.2×, Student-t 6.4×,
  Poisson 6.9×, Binomial 12.3×, Beta 17.4×. Quantile: Student-t 2.9×, Poisson 5.7×,
  Binomial 10.3×, Gamma 10.5×, Beta 16.5×.
- Student-t batch CDF is level (1.03×) while its scalar CDF is 6.4×.
- Raw output: `bench_{dist,elem}_{main,dev}_pass{1,2,3}.txt`.

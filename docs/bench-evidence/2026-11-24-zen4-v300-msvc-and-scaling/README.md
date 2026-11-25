# Z: the all-MSVC configuration and corvus per-call scaling, for #156 (2026-11-24)

Two additions to Z's L6 statement on #156. The clang-cl record is
`../2026-11-24-zen4-v300-l4/`.

## Conditions

- Asus TUF A16, Ryzen 7 7445HS (6 cores / 12 threads, AVX-512), Windows 11 Pro
  10.0.26300, power scheme Performance, on AC, screen locked.
- `dev/v3.0.0-corvus-adoption` at `ec4ed8c` (library code as at `e4026b3`,
  corvus v1.2.0) against `main` at `c049f23` (code `7069adc`).
- One detached script (`quiet.ps1.txt`, phase `i156`); `run.txt` is its log.
  18 runs, each launched directly and opted out of Windows power throttling,
  with `LIBSTATS_BENCH_WARMUP_SECONDS=25`. CPU 3.0–7.4% at the gates; the
  10-s clock probe read 127.0–127.5 ms per pass before each group and after
  the last (boost is 127–130, base clock about 190). 19:32–19:51 local.
- `corvus_scaling_bench.cpp` gains `erfinv` and `erfcinv` columns in this
  change, and its `getenv` is replaced on Windows.

## 1. The all-MSVC configuration

What a default Windows build gets: libstats and the fetched corvus both
compiled by `cl.exe` 19.52 (`configure_msvc_excerpt.txt`). corvus then
dispatches AVX2 under Highway's MSVC blocklist while libstats' own kernels
take `/arch:AVX512`. Benches built with `cl.exe` and the flags of
`tools/bench/README.md` against each tree's MSVC `stats_static.lib`
(`build_bench.ps1.txt`). Three passes, `main` first on passes 1 and 3, dev
first on pass 2, sustained timer.

Median of three passes, `main` → dev (dev/main), ns/element
(`compare_msvc.txt`):

| ns/elem | pdf@1e6 | lpdf@1e6 | cdf@1e6 | pdf-sc | lpdf-sc | cdf-sc | quantile |
|---|---|---|---|---|---|---|---|
| gamma | 1.0 → 1.0 (1.00×) | 0.7 → 0.7 (1.00×) | 25.5 → 35.4 (1.39×) | 17.2 → 17.2 (1.00×) | 11.5 → 11.4 (0.99×) | 113.9 → 455.7 (4.00×) | 549.1 → 5853.5 (10.66×) |
| beta | 0.8 → 0.8 (1.00×) | 0.7 → 0.7 (1.00×) | 11.2 → 145.0 (12.95×) | 22.1 → 22.1 (1.00×) | 16.3 → 16.3 (1.00×) | 119.3 → 2238.4 (18.76×) | 673.3 → 13036.2 (19.36×) |
| poisson | 9.7 → 5.1 (0.53×) | 8.1 → 4.1 (0.51×) | 18.9 → 63.2 (3.34×) | 46.3 → 26.0 (0.56×) | 39.1 → 20.7 (0.53×) | 83.6 → 601.3 (7.19×) | 329.7 → 1773.5 (5.38×) |
| student_t | 0.7 → 0.7 (1.00×) | 0.5 → 0.5 (1.00×) | 21.9 → 21.8 (1.00×) | 16.9 → 17.0 (1.01×) | 11.2 → 11.2 (1.00×) | 164.3 → 1251.0 (7.61×) | 831.5 → 3057.5 (3.68×) |
| binomial | 19.2 → 12.8 (0.67×) | 17.6 → 10.7 (0.61×) | 31.3 → 191.2 (6.11×) | 89.2 → 61.1 (0.68×) | 79.6 → 51.3 (0.64×) | 140.6 → 2304.9 (16.39×) | 507.6 → 6883.6 (13.56×) |
| gaussian | 0.3 → 0.3 (1.00×) | 0.1 → 0.1 (1.00×) | 3.6 → 3.4 (0.94×) | 8.3 → 8.4 (1.01×) | 6.2 → 6.2 (1.00×) | 22.6 → 22.8 (1.01×) | 74.8 → 112.2 (1.50×) |
| lognormal | 0.5 → 0.5 (1.00×) | 0.4 → 0.4 (1.00×) | 1.6 → 1.4 (0.87×) | 14.3 → 14.3 (1.00×) | 9.1 → 9.1 (1.00×) | 21.6 → 20.0 (0.93×) | 81.9 → 122.5 (1.50×) |
| von_mises | 1.5 → 1.4 (0.93×) | 0.9 → 0.8 (0.89×) | 32.2 → 31.1 (0.97×) | 16.4 → 16.6 (1.01×) | 12.6 → 12.7 (1.01×) | 143.2 → 143.1 (1.00×) | 395.7 → 423.2 (1.07×) |
| weibull | 0.8 → 0.8 (1.00×) | 0.6 → 0.6 (1.00×) | 2.3 → 2.3 (1.00×) | 21.4 → 21.4 (1.00×) | 14.4 → 14.4 (1.00×) | 25.9 → 25.8 (1.00×) | 26.6 → 26.6 (1.00×) |
| exponential | 0.3 → 0.3 (1.00×) | 0.1 → 0.1 (1.00×) | 0.6 → 0.6 (1.00×) | 8.1 → 8.4 (1.04×) | 5.8 → 5.8 (1.00×) | 9.6 → 9.4 (0.98×) | 10.9 → 10.9 (1.00×) |

| ns/elem | exp | log | cos | sin | erf |
|---|---|---|---|---|---|
| n=1000 | 0.68 → 0.68 (1.00×) | 0.83 → 0.83 (1.00×) | 1.11 → 1.11 (1.00×) | 1.12 → 1.11 (0.99×) | 4.40 → 3.35 (0.76×) |
| n=1000000 | 0.68 → 0.68 (1.00×) | 0.83 → 0.83 (1.00×) | 1.11 → 1.11 (1.00×) | 1.12 → 1.12 (1.00×) | 4.43 → 3.34 (0.75×) |

The largest max/min over the three passes is 1.17 for any cell above 0.2 ns;
one cell printed as 0.1, 0.2, 0.1 gives the 2.00 in `compare_msvc.txt`.

The same ratios under the two Windows builds, dev/main:

| | batch CDF, clang-cl | batch CDF, MSVC | scalar CDF, clang-cl | scalar CDF, MSVC | quantile, clang-cl | quantile, MSVC |
|---|---|---|---|---|---|---|
| gamma | 0.85× | 1.39× | 3.80× | 4.00× | 10.2× | 10.7× |
| beta | 5.71× | 12.95× | 13.7× | 18.8× | 15.6× | 19.4× |
| poisson | 2.02× | 3.34× | 7.23× | 7.19× | 5.50× | 5.38× |
| student_t | 1.00× | 1.00× | 5.39× | 7.61× | 2.81× | 3.68× |
| binomial | 3.04× | 6.11× | 13.3× | 16.4× | 10.6× | 13.6× |

- **Batch CDF is where MSVC costs most**: Beta 60 → 145 ns/element and
  Binomial 86 → 191 against clang-cl, about 2.2–2.4× more, which is the
  AVX2-against-AVX-512 gap in corvus's per-element cost (section 2).
- Scalar CDF and quantile are closer between the two builds (1.0–1.4×): a
  single call costs about one vector either way (section 2).
- Densities, the discrete pmfs (0.51–0.68×) and the elementary kernels read
  as under clang-cl. `erf` is 0.75× here against 0.47×.
- Gaussian and LogNormal quantile are 1.50× slower, as under clang-cl.

## 2. corvus per call against span length

`corvus_scaling_bench`, corvus called directly, ns per element, median of
three passes (`scaling_{clangcl,msvc}_pass{1,2,3}.txt`; the largest max/min
over the passes of any cell is 1.26).

**clang-cl build, corvus `AVX3_ZEN4`:**

| n | erf | exp | lgamma | gamma_p | beta_p | gamma_p_inv | erfinv | erfcinv |
|---|---|---|---|---|---|---|---|---|
| 1 | 32.6 | 35.7 | 88.8 | 416.3 | 1874.1 | 4781.3 | 130.4 | 131.8 |
| 2 | 16.2 | 17.8 | 63.0 | 434.5 | 1031.9 | 4527.8 | 79.3 | 79.7 |
| 4 | 8.0 | 9.2 | 31.6 | 104.0 | 516.2 | 2506.3 | 39.6 | 39.9 |
| 8 | 3.7 | 3.8 | 20.3 | 107.7 | 417.4 | 1171.8 | 19.4 | 19.6 |
| 32 | 2.1 | 1.9 | 16.5 | 106.0 | 329.0 | 1095.0 | 16.4 | 16.6 |
| 256 | 1.7 | 1.4 | 15.2 | 86.2 | 355.9 | 1145.2 | 15.7 | 15.9 |
| 4096 | 1.7 | 1.7 | 16.1 | 93.6 | 365.5 | 1165.8 | 15.6 | 15.8 |
| 65536 | 1.6 | 1.3 | 16.0 | 93.2 | 365.0 | 1158.3 | 15.5 | 15.7 |

**MSVC build, corvus `AVX2`:**

| n | erf | exp | lgamma | gamma_p | beta_p | gamma_p_inv | erfinv | erfcinv |
|---|---|---|---|---|---|---|---|---|
| 1 | 23.7 | 21.9 | 88.8 | 444.3 | 2399.6 | 5077.5 | 122.3 | 120.2 |
| 2 | 11.8 | 10.8 | 66.8 | 433.4 | 1278.3 | 4682.7 | 85.6 | 76.8 |
| 4 | 5.2 | 4.3 | 33.2 | 110.3 | 638.4 | 2624.7 | 42.4 | 38.2 |
| 8 | 4.1 | 3.3 | 28.6 | 214.6 | 863.2 | 1915.0 | 41.1 | 36.9 |
| 32 | 3.5 | 2.9 | 38.3 | 187.9 | 712.8 | 2156.1 | 40.8 | 36.7 |
| 256 | 3.3 | 2.7 | 37.3 | 161.5 | 766.5 | 2145.5 | 38.9 | 34.8 |
| 4096 | 3.2 | 2.7 | 37.2 | 165.4 | 800.1 | 2180.9 | 38.1 | 33.9 |
| 65536 | 3.2 | 2.7 | 36.8 | 165.7 | 797.1 | 2175.2 | 38.3 | 34.1 |

Reading, for the scalar routes:

- **A one-element call costs about one vector.** On the AVX-512 build a
  single `erfinv` or `erfcinv` call is 130 ns, and eight elements at the long-span
  rate are 124–126 ns. `lgamma` 89 against 128, `beta_p` 1874 against 2920,
  `gamma_p_inv` 4781 against 9270: between half a vector and a whole one. An
  entry point that removes only span construction and dispatch would recover
  little of this.
- **What a scalar code path at the per-element rate would be worth**, set
  beside `main`'s whole scalar call under clang-cl: `gamma_p` 93 ns against
  Gamma's scalar CDF of 108 on `main` (411 on dev); `beta_p` 365 against
  Beta's 118 (1612 on dev); `gamma_p_inv` 1158 against Gamma's quantile of
  542 (5534 on dev). So even at the per-element rate Beta's CDF and Gamma's
  quantile stay 2–3× above `main`: that part is the accuracy trade, not the
  call shape.
- **AVX2 under MSVC costs 1.8–2.4× more per element** at long spans (`gamma_p`
  166 against 93, `beta_p` 797 against 365, `gamma_p_inv` 2175 against 1158,
  `erfinv` 38 against 15.5) and about the same per single call.
- The n = 1 and n = 2 `gamma_p` cells repeat one or two fixed arguments, so
  they show one branch of the kernel, not an average.

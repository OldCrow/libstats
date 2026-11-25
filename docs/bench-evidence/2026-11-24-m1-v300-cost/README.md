# M at 92ec8e1: v3.0.0 (corvus v1.2.0) against main, for #156, 2026-11-24

Mac Mini M1 (NEON, 2 lanes; 4 performance + 4 efficiency cores), macOS 27.2.1,
AppleClang 21.0.0. `dev/v3.0.0-corvus-adoption` at `92ec8e1` against `main` at
`c049f23`, both fresh Release builds. Why: `PLAN.md`, "#156 — what each machine
still owes", measurements A and B. This replaces M's 2026-11-21 run (old timer,
not quiet throughout, never stated on #156).

## Conditions

- Benches from the dev checkout at `92ec8e1` (sustained timer, #320), compiled
  with the library's own flags, `-O3 -DNDEBUG -std=c++20 -arch arm64
  -stdlib=libc++ -DLIBSTATS_HAS_NEON=1` (`build_bench.sh.txt`).
- Screen locked, AC power, Backblaze paused. Every binary waited for ambient
  CPU (100 − `top` idle) below 8% over two 10 s windows; each started at
  2.4–3.9% (`run.txt`; misses in `gate_samples.txt`).
- Fourteen of the user's background processes (media analysis, Spotlight,
  sync) were suspended (SIGSTOP) for each group of runs and resumed after it
  [user's decision]; the list is in `quiet_run.sh.txt` and `environment.txt`.
- The directory carries the local date; the runs are 02:44–03:23 UTC on
  2026-11-25.
- `environment.txt` reads `dirty=1` for dev: `PLAN.md` only. The #107 edits to
  `src/` were made in the working tree while the sitting ran and were not
  built; the binaries are `92ec8e1` (hashes in `environment.txt`).

## A — the bench against `main`

Three passes, `main` first on passes 1 and 3, dev first on pass 2. Median of
three, `main` → dev (dev/`main`). The largest max/min over the passes is 1.08,
leaving out one cell that prints as 0.2 or 0.3 (Exponential LogPDF at 1e6).
Columns: batch at 1e6 under AUTO, scalar at 1e5, quantile at 1e4.

| ns/elem | pdf@1e6 | lpdf@1e6 | cdf@1e6 | pdf-sc | lpdf-sc | cdf-sc | quantile |
|---|---|---|---|---|---|---|---|
| gamma | 0.9 → 1.0 (1.11×) | 0.7 → 0.7 (1.00×) | 18.5 → 46.9 (2.54×) | 22.3 → 22.3 (1.00×) | 18.1 → 18.1 (1.00×) | 123.0 → 435.5 (3.54×) | 599.8 → 5271.7 (8.79×) |
| beta | 1.4 → 1.4 (1.00×) | 1.2 → 1.2 (1.00×) | 11.9 → 147.3 (12.38×) | 49.3 → 49.3 (1.00×) | 24.7 → 24.8 (1.00×) | 97.1 → 1393.5 (14.35×) | 773.3 → 9695.3 (12.54×) |
| poisson | 6.6 → 4.3 (0.65×) | 5.0 → 3.8 (0.76×) | 15.3 → 77.3 (5.05×) | 45.9 → 30.7 (0.67×) | 32.0 → 26.2 (0.82×) | 93.4 → 521.1 (5.58×) | 327.6 → 1553.5 (4.74×) |
| student_t | 1.0 → 1.0 (1.00×) | 0.8 → 0.8 (1.00×) | 20.5 → 20.7 (1.01×) | 24.1 → 24.2 (1.00×) | 19.5 → 19.8 (1.02×) | 146.4 → 734.8 (5.02×) | 919.3 → 2115.0 (2.30×) |
| binomial | 13.4 → 11.4 (0.85×) | 11.2 → 9.9 (0.88×) | 28.3 → 187.6 (6.63×) | 80.7 → 69.4 (0.86×) | 66.6 → 59.6 (0.89×) | 156.5 → 1495.6 (9.56×) | 522.7 → 4152.0 (7.94×) |
| gaussian | 0.4 → 0.4 (1.00×) | 0.2 → 0.2 (1.00×) | 2.0 → 2.0 (1.00×) | 17.5 → 17.6 (1.01×) | 17.5 → 17.5 (1.00×) | 31.3 → 31.5 (1.01×) | 110.2 → 82.0 (0.74×) |
| lognormal | 0.9 → 0.9 (1.00×) | 0.7 → 0.7 (1.00×) | 1.2 → 1.2 (1.00×) | 21.9 → 21.9 (1.00×) | 17.6 → 17.5 (0.99×) | 38.7 → 37.0 (0.96×) | 138.3 → 91.5 (0.66×) |
| von_mises | 1.2 → 1.2 (1.00×) | 0.9 → 0.9 (1.00×) | 28.8 → 28.8 (1.00×) | 26.2 → 26.2 (1.00×) | 21.5 → 21.5 (1.00×) | 154.2 → 155.9 (1.01×) | 510.0 → 480.0 (0.94×) |
| weibull | 1.4 → 1.4 (1.00×) | 1.2 → 1.2 (1.00×) | 1.5 → 1.5 (1.00×) | 32.1 → 32.2 (1.00×) | 26.9 → 26.9 (1.00×) | 45.9 → 45.9 (1.00×) | 33.2 → 33.2 (1.00×) |
| exponential | 0.4 → 0.4 (1.00×) | 0.2 → 0.2 (1.00×) | 0.7 → 0.7 (1.00×) | 17.5 → 17.5 (1.00×) | 17.5 → 17.4 (0.99×) | 18.0 → 18.0 (1.00×) | 20.0 → 20.0 (1.00×) |

`elementary_bench` (largest max/min 1.02):

| ns/elem | exp | log | cos | sin | erf |
|---|---|---|---|---|---|
| n=1000 | 0.94 → 0.96 (1.02×) | 1.64 → 1.64 (1.00×) | 2.40 → 2.40 (1.00×) | 2.41 → 2.41 (1.00×) | 1.37 → 1.37 (1.00×) |
| n=1000000 | 0.95 → 0.94 (0.99×) | 1.64 → 1.64 (1.00×) | 2.31 → 2.33 (1.01×) | 2.40 → 2.40 (1.00×) | 1.37 → 1.37 (1.00×) |

1. **The cost is in the CDFs and quantiles of the five families that reach
   `gamma_p`, `beta_p` and their inverses.** Batch CDF: Gamma 2.5×, Poisson
   5.1×, Binomial 6.6×, Beta 12.4×; Student's t level. Scalar CDF 3.5× to
   14.4× (Beta). Quantile 2.3× (Student's t) to 12.5× (Beta).
2. **Against #156's original NEON targets:** Gamma batch CDF 46.9 ns per
   element (target `gamma_p` ≤ 40), Beta 147.3 (`beta_p` ≤ 30), and a scalar
   CDF call 435 to 1496 ns (single-lane call ≤ 250). None is met, as corvus's
   comment of 2026-10-30 said.
3. **Everything else is level or faster.** Densities are level but for the
   discrete pmfs, which gain (Poisson 0.65–0.82×, Binomial 0.85–0.89×).
   Gaussian and LogNormal quantiles gain (0.74×, 0.66×). The elementary
   kernels are level within 2%: NEON keeps its own table kernels.
4. **The old-timer run of 2026-11-21 read the same.** Its ratios (batch CDF
   2.5× / 5.0× / 6.5× / 11.7×, scalar CDF up to 14.4×, quantile up to 12.6×)
   agree with these, so the per-cell clock effect that the sustained timer
   removes on I does not show on M1 at these ratios.

## B — corvus per call against span length

`corvus_scaling_bench`, three passes; it prints `corvus NEON`. ns per element,
min of 9 runs in each pass, median of the three passes. The largest max/min
over the passes of any cell is 1.12.

| n | erf | exp | lgamma | gamma_p | beta_p | gamma_p_inv | erfinv | erfcinv |
|---|---|---|---|---|---|---|---|---|
| 1 | 9.5 | 9.1 | 48.3 | 410.8 | 1571.1 | 4371.2 | 76.3 | 77.1 |
| 2 | 5.2 | 5.0 | 26.9 | 376.2 | 860.8 | 4351.7 | 53.2 | 53.8 |
| 4 | 3.1 | 3.3 | 24.8 | 205.3 | 801.1 | 3003.6 | 27.6 | 28.2 |
| 8 | 2.4 | 2.6 | 23.0 | 295.0 | 1011.2 | 2910.7 | 38.3 | 39.1 |
| 32 | 2.1 | 2.4 | 29.4 | 275.8 | 869.7 | 3412.0 | 35.9 | 36.9 |
| 256 | 2.0 | 2.3 | 26.9 | 251.6 | 928.6 | 3290.4 | 38.7 | 39.3 |
| 4096 | 2.0 | 2.3 | 27.4 | 254.3 | 957.2 | 3337.6 | 36.9 | 37.9 |
| 65536 | 2.0 | 2.3 | 28.1 | 254.5 | 954.5 | 3357.4 | 37.7 | 38.7 |

1. **At 2 lanes a one-element call costs a whole vector for `erf`, `exp`,
   `lgamma` and `beta_p`** (a call of two costs 1.09–1.11 times a call of
   one), 0.7 of one for `erfinv` and `erfcinv` (1.40) and about half for
   `gamma_p` and `gamma_p_inv` (1.83, 1.99). The inputs differ per element and
   the iterative kernels' cost follows them, as on K.
2. **Against the per-element rate at long spans, a one-element call costs
   1.3–2.0 elements** for the six costly kernels (`gamma_p_inv` 1.3, `gamma_p`
   1.6, `beta_p` 1.6, `lgamma` 1.7, `erfinv` and `erfcinv` 2.0) and 4.0–4.8
   for `exp` and `erf`, where the call itself is most of the time. K, 4 lanes:
   2.2–3.5. Z, 8 lanes: more again. The penalty of a one-element call shrinks
   with the lane count, so the 2-lane tier has the least to recover.
3. **The library's scalar paths cost what one such call costs** (A above):
   Gamma scalar CDF 435 ns against `gamma_p` at 411; Beta scalar CDF 1394
   against `beta_p` at 1571; Gamma quantile 5272 against `gamma_p_inv` at
   4371; Gaussian quantile 82 against `erfcinv` at 77.
4. **What a scalar entry point at the per-element rate could recover:** Gamma
   scalar CDF from 435 to about 270 ns (`main`: 123), Beta scalar CDF from
   1394 to about 850 (`main`: 97), Gamma quantile from 5272 to about 4000
   (`main`: 600). So 1.3–1.6× at most, and the three would still be 2.2×,
   8.7× and 6.7× above `main`. On NEON the gap is the kernels' accuracy
   trade, not the call shape. That is an upper bound: it assumes a one-lane
   kernel as fast per element as the 2-lane one.

## Also in this sitting: the pools at four and eight threads

`pool_wake_bench --init --ws --local-pool 4,8 --gaps 0,2000`
(`wake_local_pool.csv`): pool measurement 8, which M had no window for on
2026-11-24. Call time in µs, back to back / after a 2 ms busy gap:

| elements | thread pool | work stealing | local, 4 threads | local, 8 threads |
|---|---|---|---|---|
| 4096 | 45 / 46 | 42 / 62 | 51 / 49 | 51 / 47 |
| 10,240 | 70 / 77 | 73 / 94 | 82 / 82 | 76 / 76 |
| 51,200 | 201 / 201 | 192 / 246 | 280 / 272 | 209 / 196 |
| 204,800 | 612 / 616 | 716 / 673 | 1011 / 999 | 615 / 621 |

- The bench's copy of the pool is faithful: eight local threads are within
  8 µs of the library's thread pool in every cell.
- Eight threads are 1.6× faster than four at 204,800 and 1.3× at 51,200,
  efficiency cores included; below that they are level.
- Work stealing is level with the thread pool back to back up to 51,200,
  slower at 204,800, and 16–57 µs slower after a gap. It earns nothing on M1.

**#324 on M1.** `std::thread::hardware_concurrency()` costs 0.38 µs per call
here (`hardware_concurrency_probe.cpp`, 2 million calls, five repeats
375–386 ns), against 5.4 µs on I's glibc. The thread pool's figures before
the fix (`../2026-11-24-m1-pool-wake/wake_init_ws.csv`: 43.9 / 71.7 / 201.7 /
621.8 µs back to back) and after it (45.1 / 70.4 / 201.2 / 612.0) agree
within 2%. M's pool-wake runs need no re-run for #324.

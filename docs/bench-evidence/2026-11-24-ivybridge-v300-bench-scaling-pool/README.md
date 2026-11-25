# Ivy Bridge — bench against `main`, corvus per-call scaling, and the pool after #324, 2026-11-24

One quiet sitting, three measurements. Why: `PLAN.md`, "#156 — what each
machine still owes" (A and B), and the re-run the I status line owed after
#324 (C).

- **A** `tools/bench/distributions_bench.cpp` and `elementary_bench.cpp`,
  `main` against this branch, three passes, sustained timer.
- **B** `tools/bench/corvus_scaling_bench.cpp`, three passes.
- **C** `tools/bench/pool_wake_bench.cpp`: plain, `--init`, and
  `--init --ws --local-pool 4,8 --gaps 0,2000`.

## Conditions

- MacBook Pro 9,1, i7-3820QM (4 cores / 8 threads, AVX, no AVX2, no FMA),
  Linux Mint 22.3, kernel 7.0.0-43, governor `schedutil`, driver
  `intel_cpufreq`, on AC, GCC 13.3.
- This branch at `92ec8e1`, clean tree, Release `libstats.a`, corvus v1.2.0.
  **corvus selects SSE4** (2 lanes).
- The `main` side: headers from `main` at `c049f23`, Release `libstats.a`
  built at `51e1390` (no code file differs between the two commits).
- Benches built with `g++ -O3 -DNDEBUG -std=c++20 -DLIBSTATS_HAS_AVX=1
  -DLIBSTATS_HAS_SSE2=1`; hashes in `environment.txt`.
- Run by the user from a terminal with nothing else open. Each of the 18
  items waited for a 1-minute load below 0.30 and started at 0.25–0.29
  (`run.txt`). All exited 0. The script is `p156_run.sh.txt`.
- The directory carries the local date; the run is 02:50–03:29 UTC on
  2026-11-25.

## A — the bench against `main`

ns per element, median of three passes, `main` → this branch. Batch is at
10^6 elements. The spread over the three passes is 0–2% in every cell of this
table except `main`'s gamma scalar CDF (7%).

| Distribution | Batch CDF | Scalar CDF | Quantile |
|---|---|---|---|
| gamma | 41.8 → 140.5 (3.36×) | 173.5 → 850.3 (4.90×) | 966 → 10,963 (11.35×) |
| beta | 69.0 → 480.1 (6.96×) | 346.7 → 3403 (9.82×) | 1925 → 20,668 (10.74×) |
| poisson | 48.3 → 278.6 (5.77×) | 199.0 → 1458 (7.33×) | 673 → 4382 (6.51×) |
| student_t | 95.7 → 64.3 (0.67×) | 457.2 → 1832 (4.01×) | 2399 → 4866 (2.03×) |
| binomial | 126.5 → 667.4 (5.28×) | 500.6 → 3938 (7.87×) | 1554 → 11,557 (7.44×) |

What moved elsewhere:

- **Faster on this branch**: binomial PDF and LogPDF, batch and scalar
  (0.49–0.55×); poisson PDF and LogPDF (0.77–0.85×); batch CDF of gaussian
  (10.2 → 6.5, 0.64×), lognormal (8.4 → 4.5, 0.54×) and von Mises (161.7 →
  103.3, 0.64×); von Mises scalar CDF (0.65×) and quantile (0.72×).
- **Unchanged within 2%**: weibull and exponential in every column, and every
  scalar PDF and LogPDF outside poisson and binomial.
- **Not a finding**: gaussian and exponential batch LogPDF read 1.20×, which
  is 1.0 against 1.2 ns with a 17% spread on this branch's side. The batch
  PDF and LogPDF cells of 1–5 ns spread up to 10% and none moves by more.
- **Elementary kernels** (`bench_elem_*`): `erf` 22.4 → 7.8 ns at n = 1000
  and 22.6 → 7.9 at 10^6 (0.35×); `exp`, `log`, `cos` and `sin` within 3%.

This replaces the statement of 2026-11-22 on #156, which was taken with the
old timer.

## B — corvus per call against span length

ns per element, min of 9 runs in each pass, median of the three passes. The
largest spread over the passes is 8% (in the n = 1 row) and 1.5–5% in the
other rows.

| n | erf | exp | lgamma | gamma_p | beta_p | gamma_p_inv | erfinv | erfcinv |
|---|---|---|---|---|---|---|---|---|
| 1 | 23.5 | 24.6 | 109.9 | 772.7 | 4122.4 | 9499.0 | 162.4 | 165.2 |
| 2 | 11.4 | 12.2 | 66.9 | 1032.2 | 2347.9 | 9442.8 | 115.6 | 117.0 |
| 4 | 9.8 | 10.3 | 58.8 | 386.4 | 2115.4 | 5762.0 | 60.3 | 62.1 |
| 8 | 8.7 | 9.2 | 59.3 | 786.1 | 2790.5 | 6354.5 | 86.9 | 88.4 |
| 32 | 7.8 | 8.1 | 72.7 | 645.0 | 2350.6 | 7250.9 | 83.2 | 84.8 |
| 256 | 11.5 | 8.3 | 64.7 | 550.4 | 2496.3 | 7092.6 | 87.0 | 88.4 |
| 4096 | 11.4 | 8.3 | 67.2 | 576.4 | 2658.6 | 7174.4 | 86.1 | 87.6 |
| 65536 | 7.8 | 7.9 | 68.0 | 576.5 | 2650.8 | 7240.4 | 86.1 | 87.8 |

1. **A one-element call costs a whole 2-lane vector for `erf` and `exp`**: a
   call of two costs 0.97–0.99 of a call of one. For the other six kernels
   the bench cannot say, because it draws fresh arguments for each n and the
   iterative kernels' cost follows the arguments (see "The n = 4 row" below).
2. **Against the per-element rate at long spans, a one-element call costs
   1.3–1.9 elements** for the six costly kernels (`gamma_p_inv` 1.3,
   `gamma_p` 1.3, `beta_p` 1.6, `lgamma` 1.6, `erfinv` 1.9, `erfcinv` 1.9)
   and 3.0–3.1 for `erf` and `exp`. Kaby Lake at 4 lanes reads 2.2–3.5 and
   6.0. The n = 1 figure is one argument draw, so these are approximate.
3. **So a scalar entry point has less to recover at 2 lanes.** Upper bound,
   from A's scalar figures divided by the factors in 2: gamma scalar CDF
   850 → about 635 ns (`main` 174), beta scalar CDF 3403 → about 2190
   (`main` 347), gamma quantile 10,963 → about 8360 (`main` 966). That
   leaves them 3.7×, 6.3× and 8.7× above `main`.
4. **Per element at long spans, SSE4 here costs 2.5–2.8× Kaby Lake's AVX2**
   for the incomplete functions (`gamma_p` 577 against 234, `beta_p` 2651
   against 937, `gamma_p_inv` 7240 against 2978) and 1.1–1.3× for `erf`,
   `exp` and `lgamma`.

### The n = 4 row

`gamma_p`, `gamma_p_inv`, `erfinv` and `erfcinv` are 1.3–2× cheaper per
element at n = 4 than at n = 2 or n = 8. This is the bench's arguments, not
the machine: the generator is seeded once and each n draws a fresh set, so
the n = 4 row is the same four arguments everywhere, and Kaby Lake and Zen 4
show the same dip (`gamma_p` 145.6 at n = 4 against 315.6 at n = 8 on K;
104 against 108–215 on Z). Rows below n = 32 compare argument draws as much
as span lengths.

### `erf` at 256 and 4096: 4K aliasing, a bench artifact

`erf` reads 11.5 ns at n = 256 and 4096 against 7.8 at 32 and 65,536, in all
three passes. Kaby Lake and Zen 4 do not show it.

**Cause** [probed 2026-11-24, machine in use, so indicative; repeated over
three runs]: at those two sizes glibc places the bench's output vector 32
bytes after the input vector modulo 4096. `erf_alias_probe.cpp.txt` holds
the input at a page-aligned address and moves the output:

| Output address past a page multiple, bytes | 0 | 16 | 32 | 48 | 64 | 4128 |
|---|---|---|---|---|---|---|
| `erf`, ns per element, n = 256 | 8.0 | 8.1 | 12.1 | 8.7 | 7.8 | 11.4 |

The 32-byte column reads 11.3–12.1 at every n from 32 to 65,536 and the
columns that do not alias 7.4–8.8 (`erf_alias_probe.txt`). So the kernel
costs about 7.5–8 ns per element on this machine and 1.5× that when a caller's output sits 32 bytes
past its input modulo a page. `exp` was not probed at that offset; in the
bench it reads 8.3 at both sizes, where it had the same two buffers.

## C — the pool after #324

**The fix is confirmed.** With the polling pool alive, the library's thread
pool now matches the bench's four-thread local pool. Call, µs, back to back,
sizes 4096 / 10,240 / 51,200 / 204,800:

| | 4096 | 10,240 | 51,200 | 204,800 |
|---|---|---|---|---|
| Thread pool before (`../2026-11-22-ivybridge-pool-wake/`) | 44.7 | 77.0 | 246 | 854 |
| Thread pool now | 34.4 | 66.8 | 232 | 842 |
| Local pool of four, now | 34.3 | 66.2 | 232 | 842 |

The last worker starts 11–15 µs after the submit at every gap against 21–30
before (`wake_init.csv`). The same holds at a 2 ms gap
(`wake_local_pool.csv`).

**Plain pool, `schedutil`** (`wake_plain.csv`), thread pool, sizes as above:

| Gap | Call, µs | Last worker starts, µs | Work, ns per element |
|---|---|---|---|
| 0 | 36 / 62 / 218 / 826 | 12 / 10 / 10 / 48 | 17.0 / 14.7 / 13.8 / 13.8 |
| 50 µs | 94 / 170 / 362 / 910 | 30 / 30 / 36 / 80 | 46.7 / 40.2 / 21.0 / 13.8 |
| 500 µs | 71 / 154 / 713 / 1908 | 48 / 55 / 86 / 94 | 14.5 / 23.5 / 42.7 / 30.1 |
| 2 ms | 78 / 153 / 634 / 2477 | 52 / 56 / 68 / 88 | 14.5 / 22.2 / 40.2 / 40.2 |
| 20 ms | 134 / 217 / 651 / 2417 | 53 / 71 / 85 / 64 | 47.3 / 40.4 / 40.2 / 40.2 |

1. **Back to back, short calls are now at the full clock.** A 4096-element
   call takes 36 µs against 111 before the fix and a 10,240-element call 62
   against 193; the work is at 17.0 and 14.7 ns per element against 40.2 and
   38.7. Finding 2 of the earlier README ("slow back to back as well, for
   short calls") no longer holds at gap 0.
2. **With any gap from 50 µs it is as before**: 94 µs at 4096 with a 50 µs
   gap, and the work at 40 ns from 2 ms at 51,200 and above. The governor
   finding stands.

Not traced: why removing about 10 µs per call keeps the workers' clock up
back to back. No frequency counter was sampled.

## Files

| File | What |
|---|---|
| `bench_dist_{main,dev}_pass{1,2,3}.txt` | A, `distributions_bench` |
| `bench_elem_{main,dev}_pass{1,2,3}.txt` | A, `elementary_bench` |
| `scaling_pass{1,2,3}.txt` | B |
| `wake_plain.csv`, `wake_init.csv`, `wake_local_pool.csv` | C |
| `run.txt`, `environment.txt`, `p156_run.sh.txt` | times and loads, build and machine, the script |
| `erf_alias_probe.cpp.txt`, `erf_alias_probe.txt` | the indicative probe and one run of it |

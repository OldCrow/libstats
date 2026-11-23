# M1 — batch-gap check (L7), 2026-11-22 and 2026-11-23

Does PARALLEL still beat VECTORIZED when batches do not arrive back to back?
`tools/bench/batch_gap_bench.cpp`, each run its own process. Why this is
measured: `PLAN.md`, "Pool wake-up and batch spacing". The Ivy Bridge run it
is compared with: `../2026-11-22-ivybridge-batch-gap/`.

## Conditions

- Mac Mini M1 (4 performance + 4 efficiency cores, NEON), macOS 27.2.1.
- `dev/v3.0.0-corvus-adoption` at `7a1ec20`, AppleClang 21.0.0, `-O3 -DNDEBUG
  -std=c++20 -arch arm64 -DLIBSTATS_HAS_NEON=1`, linked against the Release
  `libstats.a` of that commit.
- Screen locked, nothing else started. Each run waited for ambient CPU
  (100 − `top` idle) below a limit over two 10 s windows: 15% in pass 1, 8% in
  pass 2 (12% after three hours).
- **No run was quiet from start to end except pass 1 `--init`.** macOS indexing
  (`mediaanalysisd`, `corespotlightd`) held one to three cores during the
  others, starting within seconds of a passed gate. `conditions.txt` has the
  load during each run. Pass 1 `--init --auto` ran at a load average of 16 and
  is not a record; pass 2 did not get a window for it before its deadline.
- Gap: serial work on the calling thread (a clock-reading loop). Only the time
  inside the batch call is counted; each cell runs 0.4 s unmeasured, then at
  least 1 s and 20 calls; median call.

## Files

| File | Pool state |
|---|---|
| `gap_plain.csv`, `pass2/gap_plain.csv` | only the thread pool, created by the first PARALLEL batch |
| `gap_init.csv`, `pass2/gap_init.csv` | `initialize_performance_systems()` first, which also starts the work-stealing pool; its idle workers poll |
| `gap_init_auto.csv` | as `gap_init.csv`, with an AUTO column (AUTO never selects work stealing on M1); loaded run |
| `fit.txt` | `scripts/fit_gap_model.py` on the four plain and `--init` CSVs |
| `conditions.txt` | run logs and the load during each run |
| `l7_gap_m_20261122.sh`, `l7_gap_r2_m_20261123.sh` | the two passes |

## Result

Smallest measured size from which PARALLEL wins at every larger size
(sizes 4096, 10000, 25000, 50000, 100000, 200000). Both passes give the same
table:

| Gap | Poisson PDF, plain | Poisson PDF, `--init` | Gaussian PDF, plain | Gaussian PDF, `--init` |
|---|---|---|---|---|
| 0 | 10000 | 4096 | 100000 | 100000 |
| 50 µs | 10000 | 4096 | 100000 | 100000 |
| 500 µs | 10000 | 4096 | 100000 | 100000 |
| 2 ms | 10000 | 4096 | 100000 | 100000 |
| 20 ms | 10000 | 4096 | 100000 | 100000 |

ns/element, pass 1 / pass 2:

| Cell | Gap | VECTORIZED plain | VECTORIZED `--init` | PARALLEL plain | PARALLEL `--init` |
|---|---|---|---|---|---|
| Poisson 4096 | 0 | 9.18 / 9.14 | 9.81 / 9.82 | 10.9 / 11.7 | 9.7 / 8.8 |
| | 2 ms | 9.17 / 9.16 | 9.83 / 9.82 | 14.5 / 11.9 | 9.3 / 9.1 |
| | 20 ms | 9.16 / 9.23 | 9.89 / 9.92 | 15.1 / 14.5 | 8.7 / 8.2 |
| Poisson 10000 | 0 | 9.16 / 9.15 | 9.83 / 9.85 | 6.75 / 6.84 | 7.12 / 6.80 |
| | 2 ms | 9.16 / 9.41 | 9.85 / 9.85 | 7.69 / 7.57 | 6.85 / 6.69 |
| | 20 ms | 9.17 / 9.16 | 9.89 / 9.90 | 7.92 / 7.97 | 7.22 / 6.77 |
| Poisson 200000 | 0 | 9.29 / 9.50 | 9.87 / 9.88 | 2.43 / 2.44 | 2.67 / 2.74 |
| | 20 ms | 9.20 / 9.28 | 9.84 / 9.85 | 2.34 / 2.33 | 2.50 / 2.51 |
| Gaussian 4096 | 0 | 1.46 / 1.47 | 1.53 / 1.53 | 4.19 / 4.23 | 6.41 / 4.46 |
| | 20 ms | 1.53 / 1.59 | 1.70 / 1.66 | 8.99 / 7.98 | 5.43 / 4.48 |
| Gaussian 100000 | 0 | 1.47 / 1.49 | 1.55 / 1.55 | 1.01 / 1.04 | 1.12 / 1.06 |
| | 2 ms | 1.45 / 1.45 | 1.56 / 1.55 | 1.17 / 1.12 | 1.00 / 0.89 |
| | 20 ms | 1.45 / 1.52 | 1.56 / 1.56 | 1.30 / 1.15 | 1.09 / 0.86 |
| Gaussian 200000 | 0 | 1.47 / 1.47 | 1.56 / 1.55 | 0.66 / 0.70 | 0.68 / 0.69 |
| | 20 ms | 1.48 / 1.51 | 1.59 / 1.56 | 0.73 / 0.78 | 0.76 / 0.73 |

What it shows on this machine:

1. **The crossover does not move with the gap**, with or without the
   work-stealing pool. This is unlike Ivy Bridge, where the plain crossover
   went from 25000 to 50000 (Poisson) and to 200000 (Gaussian).
2. **A gap does slow plain PARALLEL, but only where the work per call is
   small**: Poisson at 4096 by 24–39% from back to back to 20 ms, at 10000 by
   17%; Gaussian at 4096 by about 2 times, at 100000 by 11–29%. Poisson from
   25000 and Gaussian at 200000 do not change. None of it reaches a crossover.
   Ivy Bridge's plain PARALLEL was slow at every gap, back to back included.
3. **With the work-stealing pool started, the gap does not matter**, as on Ivy
   Bridge. What that buys here is smaller: Poisson at 4096 becomes a tie with
   plain VECTORIZED (8.2–9.7 against 9.2), Poisson at 10000 gains 9–15% at
   gaps of 500 µs and more, Gaussian at 100000 gains 15–26% at 2 ms and 20 ms.
   Ivy Bridge gained up to 3 times.
4. **The polling costs the serial path about 7%**: VECTORIZED reads a median
   1.07 times slower in the `--init` run (Poisson 1.07 in both passes,
   Gaussian 1.07 and 1.06), at every size and gap. Ivy Bridge: 1.13.
5. **The polling also costs large Poisson PARALLEL batches about 11%**
   (median 1.11 and 1.12 at 100000 and 200000). Gaussian at 200000 is
   unchanged. Ivy Bridge showed no such cost.
6. **The process used about 1.65 cores with the pool started and 1.0 without**
   (`conditions.txt`; one core is the gap loop). That is the bench's whole CPU
   read once a minute or every 20 s, not a measurement of an idle pool.
7. **The kNeon Poisson PDF threshold of 4096 is right only with the pool
   started.** Without it PARALLEL at 4096 is 1.18–1.65 times slower than
   VECTORIZED, and 10000 is the crossover.
8. AUTO in the loaded `--init --auto` run follows VECTORIZED for Gaussian to
   50000 and PARALLEL from 100000, and PARALLEL for Poisson from 4096, as the
   table says. Its times are not a record.

9. **`scripts/fit_gap_model.py` (`fit.txt`): where the plain run fits a
   model it is a start delay, and a short one.** Pass 1 reads as a delay at
   2 ms and 20 ms for both distributions, W = 18–24 µs (Gaussian) and 5–9 µs
   (Poisson; 18–19 µs in the two-parameter fit); pass 2 gives 9–16 µs for
   Gaussian and no model for Poisson within 10%. Kaby Lake: 61–89 µs. The
   loaded runs make the residuals large (up to 20% plain), so W is an order
   of magnitude here and not a constant to derive thresholds from.

Not measured here: `pool_wake_bench` (start delay and run speed per worker), which cores the woken workers run on, why the polling slows
the calling thread, an idle pool's CPU on its own, gaps by sleeping instead of
working, and any distribution but these two. The differences from Ivy Bridge
are reported and not traced (`PLAN.md`, L7).

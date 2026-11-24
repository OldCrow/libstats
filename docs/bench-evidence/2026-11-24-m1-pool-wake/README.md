# M1 — pool wake-up: a late start or a slow run? 2026-11-24

`tools/bench/pool_wake_bench.cpp`, six runs, each its own process, and an
idle-burn probe. The library's pools run a fixed integer workload (no memory,
no SIMD) in 1024-element slices; each slice records its thread and its start
and end. Why: `PLAN.md`, "Pool wake-up and batch spacing", measurements 2, 5,
6 and 7. Compared with `../2026-11-23-kabylake-pool-wake/` (K) and
`../2026-11-22-ivybridge-pool-wake/` (I).

## Conditions

- Mac Mini M1 (4 performance + 4 efficiency cores), macOS 27.2.1, AppleClang
  21.0.0, `dev/v3.0.0-corvus-adoption` at `03cb724` (library code `7a1ec20`),
  Release `libstats.a`, `-O3 -DNDEBUG -std=c++20 -arch arm64
  -DLIBSTATS_HAS_NEON=1`.
- Screen locked, Backblaze paused. Each run waited for ambient CPU
  (100 − `top` idle) below 8% over two 10 s windows.
- **`mediaanalysisd` and `corespotlightd` were suspended (SIGSTOP) before each
  gate and resumed after the run** [user's decision]: they held one to two
  cores almost without pause, and an earlier attempt without this got one run
  in 6.5 hours. `wake_m_20261124.sh`; `conditions.txt` has the log and the
  load during each run.
- Quiet during the run: all but `wake_init.csv`, where `contactsd` took up to
  a core for part of it. Its numbers agree with the same rows of
  `wake_init_ws.csv`, taken quiet.
- **Not run**: `--local-pool` (measurement 8) — no window before the deadline.
- `idle_burn.txt` was taken by hand at 08:35 with the screen in use, ungated;
  it is the process's own CPU time (`getrusage`), two runs. `idle_burn.cpp`
  is the probe (I's was not committed; it prints the same line).
- **The library code predates #324** (thread count computed once, merged at
  `f6e748e`). On I that lookup cost about 10 µs per THREAD_POOL call; what it
  costs on M1 is not measured, so the THREAD_POOL call times here may carry
  it. Differences between gaps and pool states do not.
- Medians over calls; the workload costs 15.0 ns per element serially.

| File | Options |
|---|---|
| `wake_plain.csv` | only the thread pool |
| `wake_init.csv` | `--init`: the polling work-stealing pool alive |
| `wake_init_ws.csv` | `--init --ws`: adds the WORK_STEALING rows |
| `wake_plain_sleep.csv` | `--sleep`, gaps 0, 2 ms, 20 ms: the caller sleeps in the gap |
| `wake_init_ws_sleep.csv` | `--init --ws --sleep`, the same gaps |
| `wake_init_ws_fine.csv` | `--init --ws`, gaps 0, 5, 10, 20, 40, 100 µs |
| `idle_burn.txt`, `idle_burn_plain.txt` | CPU time of a process idle for 10 s, with and without `initialize_performance_systems()` |

## Result

Thread pool, plain. Sizes 4096 / 10,240 / 51,200 / 204,800; call time in µs:

| Gap | Caller busy | Caller asleep | Serial, caller asleep |
|---|---|---|---|
| 0 | 51.7 / 80.5 / 207 / 589 | 51.9 / 81.7 / 207 / 610 | 61.4 / 153 / 770 / 3076 |
| 50 µs | 60.3 / 99.9 / 212 / 588 | | |
| 500 µs | 50.2 / 78.4 / 201 / 589 | | |
| 2 ms | 51.5 / 78.7 / 200 / 591 | 110 / 211 / 751 / 1777 | 203 / 506 / 2401 / 3535 |
| 20 ms | 86.3 / 120 / 230 / 616 | 128 / 230 / 774 / 2434 | 203 / 506 / 2529 / 6397 |

With `--init` (the polling pool alive), the same cells:

| Gap | Thread pool, caller busy | Thread pool, caller asleep | Work stealing, caller busy | Work stealing, caller asleep |
|---|---|---|---|---|
| 0 | 43.9 / 71.7 / 202 / 622 | 46.4 / 72.4 / 214 / 657 | 42.7 / 73.5 / 193 / 640 | 42.9 / 74.0 / 187 / 655 |
| 2 ms | 47.0 / 78.2 / 201 / 618 | 45.8 / 76.4 / 222 / 698 | 67.8 / 93.0 / 232 / 651 | 40.3 / 73.4 / 233 / 700 |
| 20 ms | 50.9 / 81.3 / 206 / 621 | 45.1 / 81.0 / 222 / 690 | 66.8 / 100 / 227 / 650 | 43.6 / 76.3 / 214 / 631 |

1. **With a busy caller a gap costs nothing up to 2 ms.** At 20 ms the call
   costs 23–40 µs more at every size, the last worker starts 7–35 µs later,
   and the work per element is unchanged. K pays 60–75 µs from 500 µs on. This
   is the start delay `fit_gap_model.py` gave for M's L7 runs (5–24 µs), and
   it explains why M's crossovers did not move.
2. **With a sleeping caller the clock drops, serial work included.** After a
   2 ms or 20 ms sleep, serial work runs 3.3 times slower (15.0 to 49.4 ns per
   element) up to 51,200 elements, recovering inside longer calls. Pool
   workers run at the same 49.4, only four of the eight take slices, and the
   call is 2.1–2.5 times longer at 4096, 3.6–3.7 times at 51,200, 2.9–4.0
   times at 204,800. K shows the same (3.0 times); it is not a pool defect.
3. **What the polling pool buys on M1 is the clock, as on K.** With it alive
   a sleeping caller loses nothing: serial 16.1 ns at every gap, thread pool
   46.4 to 45–46 µs at 4096, 214 to 222 at 51,200.
4. **Its price**: serial integer work 7% (14.98 to 16.08 ns per element, the
   figure L7 gave for VECTORIZED); thread-pool calls at 204,800 5–10%
   (589 to 622–651 µs); and 5.6–6.6 s of CPU per 10 s idle (2.1 user, 3.5–4.5
   system), against 0.00 with no pool. I: 24.8 s. Back to back at 4096 and
   10,240 the thread pool is 10–15% faster with the polling pool alive
   (51.7 to 44–47 µs), where K's is 21–27% slower.
5. **Work stealing itself earns nothing here.** Back to back it matches the
   thread pool within 5% at every size (both run eight workers on M1; K's
   thread-count difference does not arise). After a busy gap it is slower:
   +12 µs at a 5 µs gap, +44 at 100 µs (4096), and 67–68 µs against 47–51 at
   2 and 20 ms; at 10,240 after 50 µs, 143 against 73. With a sleeping caller
   it is 2–6 µs faster at 4096 and level elsewhere. AUTO never selects it on
   M1.
6. **Efficiency cores.** Slices run at 16.0 or 23.2 ns per element (1.45
   times), the two core types. From 51,200 elements all eight workers take
   slices and the first slice to start is usually an efficiency-core one. The
   thread pool's last worker starts 29–65 µs after the submit back to back,
   so at 4096 only three workers take a slice. After a 20 ms busy gap the
   first slice reads 36–38 ns at 51,200 and 204,800: an efficiency core at a
   reduced clock, for that slice only.
7. **The 7% on serial work is the caller running at the workers' speed**:
   16.08 ns with the polling pool alive, and pool workers run at 16.03–16.11
   in every run, plain included. Why the caller alone reaches 14.98 is not
   measured.

## For the design [M's reading; undecided]

- The polling pool cannot simply be guarded off on M1. A caller that blocks
  between batches would lose 3.3 times on serial and parallel work alike.
- What it buys is one thing: the performance cluster kept awake. It pays for
  that with eight polling workers (0.6 of a core idle, 7% on serial work) and
  a second pool that AUTO never uses here. Whether one mechanism can hold the
  clock for less is a design question for all three macOS and Linux machines,
  not an M1 guard.
- For a busy caller M1 needs no spaced threshold: the delay is 23–40 µs and
  starts beyond 2 ms. The kNeon Poisson PDF threshold of 4096 follows the
  pool state (`../2026-11-22-m1-batch-gap/`), not the gap.

Not measured here: frequencies and which cluster each worker ran on
(`powermetrics`, measurement 3, needs the user), the local pools
(measurement 8), and gaps between 2 and 20 ms, where the busy-caller delay
sets in.

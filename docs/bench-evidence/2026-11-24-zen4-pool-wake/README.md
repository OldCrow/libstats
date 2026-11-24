# Zen 4 — pool wake-up, sleeping caller, idle burn and clocks, 2026-11-24

`PLAN.md`, "Measurements this calls for", items 1, 2, 3, 5 and 7 on Z, with
items 6 and 8 as M ran them. `tools/bench/pool_wake_bench.cpp` drives the
library's pools with an integer workload and records when each worker starts
and how fast it runs.

## Conditions

- Asus TUF A16, Ryzen 7 7445HS (6 cores / 12 threads, AVX-512), Windows 11 Pro
  10.0.26300, power scheme Performance, on AC, screen locked.
- `dev/v3.0.0-corvus-adoption` at `3662ae7` (last code change `6e248a0`, the
  bench timing header; library code as at `e4026b3`). clang-cl 22.1.3, the
  library's own flags, linked against the Release `stats_static.lib`.
- One detached script (`quiet.ps1.txt`, phases `wake` and `clocks`); `run.txt`
  is its log. Before each timed run: total CPU below 8% (2.9–6.8% at the
  gates; two gates waited once), a 10-s clock probe (126.9–128.2 ms per pass,
  one first reading of 133.1; boost is 127–130, base clock about 190), a 25-s
  discarded pass of the same binary. Every process was launched directly and
  opted out of Windows power throttling. `noise.txt`: the three busiest other
  processes once a minute never sum above 0.04 of a core.
- `idle_burn.cpp` is M's probe with `GetProcessTimes` in place of `getrusage`.
- **The `clocks/` run is not a timing record**: its sampler reads 36
  performance counters at 10 Hz from the same machine.

## What the runs show

**1. The thread pool's cost after a gap is a late start, and it is small.**
Busy caller, `wake_plain.csv` and `wake_init.csv`: against back to back, a
call costs 4–26 µs more at a 2 ms gap and 24–48 µs more at 20 ms; the last
worker starts 4–22 µs and 20–27 µs later; the work time per element is
unchanged (×1.00–1.02). Nothing moves below 500 µs. K read 60–75 µs from
500 µs. This accounts for L7 (`../2026-11-24-zen4-batch-gap/`): Gaussian at
50000 costs 26 µs more per call at 2 ms there, Poisson at 10000 costs 20 µs.

**2. `--init` does not remove it.** With the work-stealing pool started the
thread pool reads the same or slightly later (20 ms: +32–48 µs). The polling
pool does not hold the clock for the thread pool's workers on Z.

**3. The workers run 1.5× slower per element than the caller, at every gap,
back to back included.** SERIAL is 10.13 ns per element; a thread-pool slice
is 15.14–15.23 ns. Six workers finish 51200 elements 3.6× sooner than the
caller alone, not 6×. `clocks/` names the likely cause (below).

**4. A sleeping caller loses its own clock.** `wake_plain_sleep.csv`: after
a 2 ms or 20 ms sleep the caller's serial work is 15.2 ns per element against
10.13 (1.5×; 12.5–12.7 at 204800, where the call is long enough to recover
part-way). The thread pool's call is 87–93 µs longer at 51200 and the first
worker starts at 49 µs instead of 5. With the polling pool alive
(`wake_init_ws_sleep.csv`) the serial figure is the same 15.2 ns: on Z the
polling pool does not hold the caller's clock either, unlike M.
`gap_plain_sleep.csv` and `gap_init_sleep.csv` (`batch_gap_bench --sleep`):
both crossovers stay where they are (Gaussian 50000, Poisson 10000), but at
50000 Gaussian PARALLEL is 0.77–0.99× VECTORIZED for a sleeping caller
against 0.64–0.66× back to back.

**5. The work-stealing pool waits 13–15 ms for a busy caller.** [Reported,
not traced.] `wake_init_ws.csv`, WORK_STEALING rows: at 51200 and 204800
elements the first worker starts 12.7–14.6 ms after the submit and the call
takes 13.1–14.9 ms, at every gap from 0 to 2 ms (the thread pool: 0.14 and
0.52 ms). The 90th percentile is 16 ms, the default Windows timer tick. At a
20 ms gap one worker does all the work at the caller's own speed
(10.16 ns per element). With a sleeping caller at 2 ms and 20 ms the same
pool starts in 15–17 µs and beats the thread pool (189–200 µs against
223–229 at 51200). The pool's idle workers yield ten times and then call
`sleep_for(100 µs)` (`src/work_stealing_pool.cpp`); on Windows that sleep
lasts until the next timer tick unless something has raised the timer
resolution, which a sleeping caller's own short sleeps would do. That
reading fits every row but was not tested.
**The batch API does not show this on Z**: in the L4 bundle forced
WORK_STEALING reads the same as forced PARALLEL (Gaussian PDF at 50000: 29.8
against 29.7 µs), and AUTO never selects work stealing here. It applies to
code that drives `WorkStealingPool` directly.

**6. Idle burn is small.** After `initialize_performance_systems()` a process
idle for 10 s uses 0.11 s user and 0.08 s system (0.19 CPU-s; I 24.8, M
5.6–6.6). With no pool started: 0.00. The same long sleeps that make finding
5 keep the polling pool cheap here.

**7. Twelve threads beat six on this workload** (`wake_local_pool.csv`,
item 8). A bench-owned pool of 6 threads matches the library's (142.7 against
143.1 µs at 51200: the copy is faithful). With 12 threads it is 90.5 µs at
51200 and 277.1 µs at 204800 against 143.1 and 520.0: 1.6× and 1.9× faster.
The library's thread pool takes the 6 physical cores. The workload is an
integer loop; an AVX-512 kernel may not gain the same from the second
hardware thread.

## Clocks and idle states (item 3, `clocks/`)

`% Processor Performance`, `Parking Status` and `% Processor Utility` per
logical processor at 10 Hz while `pool_wake_bench` runs (10240, 51200 and
204800 elements) back to back, at 2 ms and 20 ms gaps, with `--init --ws` at
20 ms, and with a sleeping caller at 20 ms. The counters read without
elevation. Nominal frequency 3201 MHz.

- **No logical processor is parked in any sample**, in any run. The gap cost
  on Z is not core parking.
- **Four logical processors boost and the rest sit at nominal.** During
  every run processors 0, 1, 8 and 9 read 130–144% of nominal; the other
  eight read 99%, including 3, 5 and 7 at 41–44% utilisation in the
  back-to-back run. 144 against 99 is the 1.5× of finding 3: the caller is
  on a boosted core and most workers are not.
- The readings do not change between gap 0, 2 ms and 20 ms. With finding 1
  (work time ×1.00–1.02), the late start after a gap is the workers' wake-up,
  not a slower clock.
- A 10 Hz sample of a utilisation-weighted counter cannot show a state that
  lasts microseconds; idle-state residency was not read.

## The two models on L7's CSVs (item 1, `fit_gap_model_l7.txt`)

`scripts/fit_gap_model.py` on `gap_plain.csv` and `gap_init.csv`. Unlike K,
Z does not read as a pure delay: Poisson fits a factor (k 1.05 at 500 µs,
1.13 at 2 ms, 1.30 at 20 ms, residual 4–18%), Gaussian fits neither (delay
14–16 µs at 2 ms and 36–42 µs at 20 ms with residuals above 1; factor
1.43–1.50 and 2.09–2.19 with residuals 0.21–0.26). The integer bench above
shows a delay of the same size and no slow run, so whatever slows a real
kernel after a gap is not the core clock that bench measures. Not
investigated.

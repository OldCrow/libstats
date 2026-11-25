# Ivy Bridge — pool wake-up under `schedutil` and under `performance`, 2026-11-23

`tools/bench/pool_wake_bench.cpp`, six runs per governor, each its own
process, plus an idle-burn probe. The library's pools run a fixed integer
workload (no memory, no SIMD) in 1024-element slices; each slice records its
thread and its start and end. Why: `PLAN.md`, "Pool wake-up and batch
spacing", measurements 2, 5, 6, 7 and 8, and the governor test proposed under
"K, measured".

Two directories, one sitting:

- this one: governor `schedutil` (the machine's default);
- `../2026-11-23-ivybridge-pool-wake-performance/`: governor `performance`,
  the same script an hour later. The user switched the governor by hand and
  restored `schedutil` afterwards.

The directory dates differ only because the first run started before local
midnight; `run.txt` has the UTC times.

## Conditions

- MacBook Pro 9,1, i7-3820QM (4 cores / 8 threads, AVX, no AVX2), Linux Mint
  22.3, kernel 7.0.0-43, driver `intel_cpufreq`, GCC 13.3,
  `dev/v3.0.0-corvus-adoption` at `761bb43`, clean tree, Release `libstats.a`
  (hashes in `environment.txt`).
- Run by the user from a terminal with nothing else open. Every item waited
  for a 1-minute load below 0.30 and started at 0.25–0.29 (`run.txt`; the load
  it shows at the end of an item is the bench's own).
- The script is `pool_wake_run.sh.txt`.
- Medians over calls. The workload costs 13.0 ns per element serially
  (14.0–14.1 with the polling pool alive).
- No frequency or idle-state counter was sampled. Clock ratios below are read
  from the workload's time per element.

| File | Options |
|---|---|
| `wake_plain.csv` | only the thread pool |
| `wake_init.csv` | `--init`: the polling work-stealing pool alive |
| `wake_init_ws.csv` | `--init --ws`: adds the WORK_STEALING rows |
| `wake_plain_sleep.csv` | `--sleep`, gaps 0, 2 ms, 20 ms: the caller sleeps in the gap |
| `wake_init_ws_fine.csv` | `--init --ws`, gaps 0, 5, 10, 20, 40, 100 µs |
| `wake_local_pool.csv` | `--init --ws --local-pool 4,8`, gaps 0 and 2 ms |
| `idle_burn.txt` | CPU time of a process idle for 10 s after `initialize_performance_systems()` |

## Result

Thread pool, plain, the caller busy in the gap. Sizes 4096 / 10,240 / 51,200 /
204,800. Serial for comparison: 58 / 133–144 / 686–709 / 2710–2720 µs under
`schedutil`, 53 / 133 / 668 / 2669 under `performance`, at every gap.

**`schedutil`**

| Gap | Call, µs | Last worker starts, µs | Work, ns per element |
|---|---|---|---|
| 0 | 111 / 193 / 337 / 943 | 55 / 59 / 59 / 114 | 40.2 / 38.7 / 17.2 / 13.8 |
| 50 µs | 132 / 201 / 526 / 956 | 62 / 56 / 108 / 110 | 40.3 / 40.3 / 28.6 / 13.8 |
| 500 µs | 84 / 196 / 752 / 2171 | 61 / 70 / 118 / 140 | 14.6 / 44.6 / 43.7 / 34.5 |
| 2 ms | 78 / 181 / 656 / 2516 | 57 / 73 / 100 / 119 | 14.6 / 40.4 / 40.2 / 40.2 |
| 20 ms | 144 / 193 / 678 / 2433 | 75 / 73 / 95 / 76 | 40.8 / 40.4 / 40.2 / 40.2 |

**`performance`**

| Gap | Call, µs | Last worker starts, µs | Work, ns per element |
|---|---|---|---|
| 0 | 41 / 66 / 226 / 836 | 19 / 19 / 20 / 56 | 15.9 / 13.8 / 13.8 / 13.8 |
| 50 µs | 39 / 66 / 227 / 843 | 17 / 18 / 20 / 59 | 16.0 / 13.8 / 13.8 / 13.8 |
| 500 µs | 78 / 103 / 255 / 865 | 60 / 63 / 63 / 64 | 14.4 / 13.8 / 13.8 / 13.8 |
| 2 ms | 79 / 111 / 259 / 865 | 60 / 60 / 60 / 61 | 16.0 / 13.8 / 13.8 / 13.8 |
| 20 ms | 84 / 104 / 275 / 867 | 65 / 56 / 55 / 62 | 14.5 / 15.1 / 14.6 / 13.8 |

1. **The slow run on this machine is the governor's policy.** Under
   `schedutil` the pool's workers run the workload at 40.2 ns per element
   against 13.8, a factor of 2.92, which is the ratio of the highest clock to
   the lowest (3.5 to 1.2 GHz). Under `performance` they run at 13.8 at every
   gap and size. The test proposed in `PLAN.md` ("If the slow run goes, it is
   the governor's policy") comes out that way.
2. **Under `schedutil` the workers are slow back to back as well, for short
   calls.** At 4096 and 10,240 elements the work is at 39–40 ns with no gap at
   all; only calls of 300 µs and more bring the clock up (17.2 ns at 51,200,
   13.8 at 204,800). From a 500 µs gap every size runs at the low clock,
   including a 2.4 ms call. The caller's own serial work stays at full speed
   throughout: it never stops.
3. **Under `performance` what remains is a start delay, as on Kaby Lake.**
   From a 500 µs gap the last worker starts 55–65 µs after the submit against
   17–20 back to back, and the call costs 28–49 µs more at every size. Kaby
   Lake measured 60–75 µs more.
4. **What that does to the crossover against serial.** Under `schedutil` with
   gaps of 500 µs and more, the pool loses at 10,240 (181–196 µs against
   133–144), is level at 51,200 and gains 1.1–1.25× at 204,800. Under
   `performance` it loses at 4096 (78–84 against 53), gains 1.2–1.3× at
   10,240, 2.4–2.6× at 51,200 and 3.1× at 204,800.
5. **A sleeping caller under `schedutil` loses the clock on everything**
   (`wake_plain_sleep.csv`). Serial work runs at 40.1 ns after a 2 ms or 20 ms
   sleep (3.1× slower), and after a 20 ms sleep it stays there through a whole
   8.3 ms call. The thread pool's last worker starts 219–225 µs after the
   submit and its work is at 40–41 ns. Under `performance` serial work is
   unchanged (13.0–13.7 ns), the pool's work is at 13.8–14.9, and the last
   worker starts 89–139 µs after the submit: the call costs 106–140 µs more
   than back to back at every size.
6. **With the polling pool alive, neither the gap nor the governor matters**
   (`wake_init.csv`, the two governors within 2% of each other in every cell
   but one). The last worker starts 21–30 µs after the submit at every gap and
   the work is at 15.6–16.1 ns. The price is in the same numbers: serial work
   8% slower (14.05 against 13.0 ns), pool work 15% slower (15.9 against
   13.8), and against the plain pool under `performance`, back to back, calls
   8% / 16% / 8% / 2% slower. Against the plain pool under `schedutil` the
   polling is a gain at every size, back to back included (45 against 111 µs
   at 4096, 854 against 943 at 204,800).
7. **Idle burn**: 24.8 s of CPU in 10 s under `schedutil` (13.1 user, 11.7
   system), 24.0 s under `performance`: two and a half cores, 37,000 voluntary
   context switches a second.
8. **Work stealing against the thread pool is eight threads against four**
   (`wake_init_ws.csv`, `wake_local_pool.csv`). WORK_STEALING uses eight
   workers from 51,200 elements and finishes 204,800 in 520–626 µs against
   853–863. A plain `ThreadPool` of eight threads does the same or better at
   every size: 34 / 55 / 165–170 / 529–531 µs against work stealing's 38–44 /
   74–94 / 178–211 / 526–626. Kaby Lake's finding 3 holds here.
9. **Work stealing's start delay after a busy gap sets in between 5 and
   10 µs** (`wake_init_ws_fine.csv`; call at 4096: 40, 46, 82, 75, 73, 71 µs
   at gaps of 0, 5, 10, 20, 40, 100 µs). Kaby Lake's onset is between 10 and
   20 µs. At 10,240 it is erratic at every gap from 10 to 50 µs (144–157 µs
   against 77–80 for the thread pool).

## Explained afterwards: the local pool's 10–15 µs

The bench's four-thread local pool is 10–15 µs faster per call than the
library's thread pool at every size (34 / 66 / 231 / 842 against 45 / 77 /
246 / 855 µs; last worker at 12–13 µs against 23–25). The row was there to
check that the bench's copy of the slice submission is faithful, and the copy
was: the library was slow.

**Cause** [traced 2026-11-23, machine in use, so indicative; the pattern
repeated exactly over three runs]: `ParallelUtils` sizes every parallel loop
with `ThreadPool::getOptimalThreadCount()`, which called
`std::thread::hardware_concurrency()` each time. On glibc 2.39 that costs
5.4 µs per call in a tight loop and about 10 µs in place between batches.

| Path, polling pool alive, back to back | Call at 4096, µs | First worker starts, µs |
|---|---|---|
| Library `parallelForSlices` on the global pool | 45.1 | 17.1 |
| The bench's copy on the same global pool | 34.4 | 7.2 |
| The bench's copy plus one `getOptimalThreadCount()` call | 45.0 | 17.0 |

**Fixed** in #324 (`main` at `7d2e046`, merged into this branch at
`f6e748e`): the count is computed once. So every THREAD_POOL figure in these
CSVs carries about 10 µs per call that the library no longer pays on this
machine: 22% of a 4096-element call, 14% at 10,240, under 2% at 204,800.
The WORK_STEALING and LOCAL_POOL rows do not carry it. **Re-measured quietly
after the fix on 2026-11-24**
(`../2026-11-24-ivybridge-v300-bench-scaling-pool/`, section C): the thread
pool now matches the local pool of four, and finding 2 above no longer holds
at gap 0. Not measured on the other machines, where the lookup may be cheap.

## Not explained

- At 4096 elements, plain, `schedutil`, gaps of 500 µs and 2 ms, only three
  workers take a slice and the work reads 14.6 ns. The cell's median slice is
  probably one the caller's side ran at full clock; not traced.
- What the 55–65 µs start delay under `performance` is. Idle-state exit is the
  candidate (`PLAN.md`, measurement 3); no counter was sampled here.

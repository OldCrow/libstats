# Kaby Lake — pool wake-up: a late start or a slow run? 2026-11-23

`tools/bench/pool_wake_bench.cpp`, six runs, each its own process. The
library's pools run a fixed integer workload (no memory, no SIMD) in
1024-element slices; each slice records its thread and its start and end.
Why: `PLAN.md`, "Pool wake-up and batch spacing", measurements 2, 5 and 6.

## Conditions

- MacBook Pro 14,1, i7-7820HQ (4 cores / 8 threads), macOS 13.7.8, AppleClang
  15, `dev/v3.0.0-corvus-adoption` at `6d95852` (library code `8c55bfc`),
  Release `libstats.a`, the bench built with the library's own flags.
- Screen locked, AC power. Each run started once the 1-minute load was below
  2.0 (1.90, 1.98, 1.94, 1.86, 1.98, 1.91; `run.txt`). Five runs came from
  `quiet_run.sh.txt`; `wake_init_ws_sleep.csv` was run by hand two minutes
  later in the same sitting, under the same gate.
- Medians over calls; the workload costs 13.7–14.1 ns per element serially.

| File | Options |
|---|---|
| `wake_plain.csv` | only the thread pool |
| `wake_init.csv` | `--init`: the polling work-stealing pool alive |
| `wake_init_ws.csv` | `--init --ws`: adds the WORK_STEALING rows |
| `wake_plain_sleep.csv` | `--sleep`, gaps 0, 2 ms, 20 ms: the caller sleeps in the gap |
| `wake_init_ws_sleep.csv` | `--init --ws --sleep`, the same gaps |
| `wake_init_ws_fine.csv` | `--init --ws`, gaps 0, 5, 10, 20, 40, 100 µs |

## Result

Thread pool, plain, the caller busy in the gap. Sizes 4096 / 10,240 / 51,200 /
204,800:

| Gap | Call, µs | Last worker starts, µs | Work, ns per element |
|---|---|---|---|
| 0 | 38.5 / 66.4 / 227 / 835 | 16 / 20 / 24 / 54 | 14.3 / 14.2 / 13.8 / 13.8 |
| 50 µs | 39.1 / 66.5 / 240 / 848 | 16 / 20 / 44 / 69 | 14.3 / 14.1 / 13.8 / 13.8 |
| 500 µs | 76.3 / 108 / 266 / 872 | 50 / 66 / 71 / 72 | 14.2 / 13.9 / 13.8 / 13.8 |
| 2 ms | 97.7 / 125 / 283 / 903 | 68 / 77 / 79 / 79 | 14.3 / 14.1 / 13.8 / 13.8 |
| 20 ms | 98.9 / 133 / 302 / 899 | 76 / 82 / 83 / 85 | 14.4 / 13.9 / 13.8 / 13.8 |

1. **With a busy caller the cost of a gap is a late start and nothing else.**
   The call costs 60–75 µs more at 20 ms than back to back at every size; the
   last worker starts 30–60 µs later; the work per element does not change
   (×1.00), nor does the caller's own serial work. At 4096 elements only two
   or three of the four workers arrive in time to take a slice. Back to back
   the last worker still starts 16–54 µs after the submit.
2. **With the polling pool alive the gap has no effect** (last worker 23–35 µs
   at every gap), and back to back the thread pool is slower: 48.8 against
   38.5 µs at 4096 (27%), 80.5 against 66.4 (21%), 247 against 227 (9%),
   858 against 835 (3%). Serial integer work is 3% slower (14.1 against
   13.65–14.0 ns).
3. **The work-stealing pool runs eight threads and the thread pool four.**
   From 51,200 elements WORK_STEALING uses seven or eight workers
   (`Workers` column) and finishes 204,800 in 535 µs against 858 (1.6×), at
   14.97 ns per element per thread against 14.2. `ThreadPool` sizes itself to
   physical cores under hyper-threading (`src/thread_pool.cpp:131`). At 4096
   and 10,240 both use four, and work stealing wins on the start: its first
   worker begins 1.3 µs after the submit against 11 (36 against 49 µs per
   call).
4. **Work stealing's start delay sets in between 10 and 20 µs of busy gap.**
   Call at 4096 elements, gaps 0 / 5 / 10 / 20 / 40 / 100 µs: 36.9 / 39.0 /
   42.9 / 58.7 / 59.5 / 75.1 µs; at 50 µs, 121 µs, with the first worker
   starting 80 µs after the submit. One cell also ran slowly: 4096 at 20 ms,
   42 ns per element.
5. **A sleeping caller, thread pool only: the clock drops, for serial work
   too.** Serial goes from 12.6 to 37.8 ns per element (3.0×) at 4096 and
   10,240 after a 2 ms or 20 ms sleep, 29–31 at 51,200 and 20–25 at 204,800:
   the clock recovers over milliseconds, inside the longer calls. The pool's
   workers run at 37.2 ns per element (2.6×), the last starts 126–141 µs
   after the submit, and the call takes 190–213 µs against 38 at 4096 (5×)
   and 1.80 ms against 0.84 at 204,800 (2.1×). Pool against serial at the
   same gap: slower at 4096 (190–213 against 156 µs), 1.4× faster at 10,240
   (2.0× back to back), 2.2× at 51,200 (2.9× back to back).
6. **A sleeping caller with the polling pool alive: nothing slows.** Serial
   14.05 to 14.3 ns; thread pool 48.6 to 54–58 µs at 4096; work stealing 36.8
   to 37.5–38.9 µs, with no start delay. Work stealing delays after a busy
   gap (4) and not after a sleeping one; that difference is not explained.

So on this machine a spaced batch pays a start delay when the caller stays
busy, and a start delay plus a 2.6–3× slow run, serial work included, when the
whole process goes idle. The polling pool removes both and costs the
back-to-back figures in 2.

Not measured: frequencies and idle states (the 3× is read from the integer
workload's time), how long the clock takes to recover, and any other machine.

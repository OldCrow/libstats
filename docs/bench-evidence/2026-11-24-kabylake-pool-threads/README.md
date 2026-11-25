# Kaby Lake — the thread pool at four and at eight threads, and the idle pool's CPU, 2026-11-24

`tools/bench/pool_wake_bench.cpp --local-pool 4,8`, three runs, each its own
process, and an idle-burn probe. Why: `PLAN.md`, "Pool wake-up and batch
spacing", measurements 7 and 8. The earlier K sitting, which this extends:
`../2026-11-23-kabylake-pool-wake/`. Ivy Bridge's run of the same:
`../2026-11-22-ivybridge-pool-wake/`.

## Conditions

- MacBook Pro 14,1, i7-7820HQ (4 cores / 8 threads), macOS 13.7.8, AppleClang
  15, `dev/v3.0.0-corvus-adoption` at `164cbce`, Release `libstats.a`, the
  bench built with the library's own flags. **The library carries #324** (the
  thread count computed once); the earlier sitting's did not.
- Screen locked, AC power. Each item started once the 1-minute load was below
  2.0 (1.87, 1.97, 1.86, then 1.94, 1.73, 1.78 for the probe; `run.txt`). The
  load it shows at the end of an item is the bench's own.
- Medians over calls. Sizes 4096 / 10,240 / 51,200 / 204,800 throughout.
- `LOCAL_POOL_n` is a `ThreadPool(n)` owned by the bench and driven by a copy
  of the library's slice submission; `THREAD_POOL` is the library's own pool
  (four threads here). The 4-thread row checks the copy.

| File | Options |
|---|---|
| `threads_plain.csv` | `--local-pool 4,8`: no polling pool |
| `threads_init_ws.csv` | `--init --ws --local-pool 4,8`: the polling pool alive, WORK_STEALING rows |
| `threads_plain_sleep.csv` | `--sleep --gaps 0,2000,20000 --local-pool 4,8`: the caller sleeps in the gap |
| `idle_burn.txt` | CPU time of a process asleep for 10 s (`idle_burn.cpp.txt`) |

## Result

Call time, µs.

**No polling pool** (`threads_plain.csv`), caller busy in the gap:

| Gap | Library pool (4) | Local, 4 threads | Local, 8 threads |
|---|---|---|---|
| 0 | 36 / 62 / 230 / 832 | 36 / 63 / 225 / 874 | 37 / 64 / 258 / 614 |
| 50 µs | 36 / 63 / 231 / 864 | 37 / 63 / 229 / 879 | 42 / 76 / 224 / 722 |
| 500 µs | 78 / 103 / 267 / 888 | 78 / 104 / 268 / 894 | 78 / 121 / 236 / 665 |
| 2 ms | 105 / 120 / 275 / 896 | 103 / 114 / 270 / 910 | 102 / 145 / 291 / 663 |
| 20 ms | 101 / 124 / 290 / 917 | 101 / 126 / 277 / 922 | 100 / 129 / 248 / 585 |

**Polling pool alive** (`threads_init_ws.csv`):

| Gap | Library pool (4) | WORK_STEALING | Local, 4 threads | Local, 8 threads |
|---|---|---|---|---|
| 0 | 47 / 77 / 245 / 854 | 36 / 59 / 161 / 543 | 45 / 78 / 241 / 870 | 45 / 81 / 225 / 511 |
| 50 µs | 46 / 76 / 242 / 847 | 102 / 146 / 226 / 508 | 46 / 77 / 241 / 854 | 46 / 78 / 265 / 505 |
| 500 µs | 48 / 78 / 243 / 847 | 89 / 125 / 215 / 542 | 47 / 77 / 241 / 863 | 47 / 78 / 265 / 518 |
| 2 ms | 49 / 79 / 240 / 847 | 92 / 120 / 219 / 561 | 48 / 79 / 242 / 856 | 49 / 81 / 294 / 499 |
| 20 ms | 52 / 82 / 245 / 852 | 90 / 154 / 216 / 574 | 50 / 84 / 243 / 858 | 49 / 84 / 263 / 503 |

1. **The bench's copy is faithful here.** The local four-thread pool is within
   about 5% of the library's in every cell of both tables, and its last worker
   starts at the same time (14–79 against 14–76 µs plain). Ivy Bridge's copy
   was 10–15 µs faster until #324; this library has #324.
2. **#324 is not separable on this machine.** Back to back, plain, the
   library pool reads 36 / 62 / 230 / 832 µs against 38 / 66 / 227 / 835 in
   the earlier sitting: 2–4 µs less at the two small sizes, level at the two
   large ones. `hardware_concurrency()` costs about 1.2 µs a call here (timed
   in a loop with the machine in use; indicative).
3. **At 204,800 elements work stealing's gain over the thread pool is the
   thread count.** With the polling pool alive, eight plain threads finish in
   499–518 µs, work stealing in 508–574, four threads in 847–870. Without the
   polling pool eight threads take 585–722: 1.2–1.5× faster than four.
4. **Below that, eight threads are not a gain.** At 4096 only four threads
   take a slice. At 10,240 seven do and the call is level with four threads
   (78–84 against 76–82 with the polling pool) or slower (121–145 against
   103–124 plain, gaps of 500 µs and more). At 51,200 it is erratic: 225–294
   against 240–245 with the polling pool, 224–291 against 230–290 plain. The
   last of eight workers starts 43–73 µs after the submit against 25–33 for
   the last of four, and each thread runs 5% slower (14.8–15.0 against 14.3 ns
   per element; two threads share a core).
5. **Back to back, work stealing beats eight plain threads below 204,800**:
   36 / 59 / 161 µs against 45 / 81 / 225. Its last worker starts 6 / 11 /
   26 µs after the submit against 19 / 47 / 68. That part of its gain is the
   spinning workers, not the thread count.
6. **With any busy gap from 50 µs, work stealing loses to both plain pools at
   4096 and 10,240** (89–154 µs against 46–84) and wins at 51,200 (215–226
   against 240–294). At 4096 with gaps of 2 ms and 20 ms its slices run at
   26.6 and 32.1 ns per element against 14.1–14.4 at shorter gaps.
7. **After a sleeping caller, eight threads do not recover the clock**
   (`threads_plain_sleep.csv`). Work per element at 204,800 is 37.9–39.2 ns on
   eight threads against 25.4–25.9 on four (12.6–15.0 back to back), so the
   call takes 1205–1221 µs against 1774–1822: a 1.5× gain where back to back
   it is 1.6×. The last of eight workers starts 168–196 µs after the submit
   against 111–122.
8. **Idle burn** (`idle_burn.txt`): after `initialize_performance_systems()`
   a sleeping process uses 10.5 s of CPU in 10 s (5.52 user, 5.00 system),
   one core. Ivy Bridge: 24.8 s, two and a half cores. With no pool, or with
   only the thread pool created, 0.00 s. macOS reports no voluntary context
   switches for the process, so that figure is absent.

## Not explained

- Why the idle polling pool costs one core here and two and a half on Ivy
  Bridge, with eight workers and the same idle loop on both.
- The 26.6 and 32.1 ns slices under work stealing at 4096 (finding 6): three
  workers, long gaps. The same shape as the caller-asleep slow run, with the
  caller busy; not traced.
- The erratic eight-thread cells at 51,200.

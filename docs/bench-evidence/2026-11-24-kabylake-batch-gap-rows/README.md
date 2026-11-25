# Kaby Lake — does one start delay predict the spaced crossover on other rows? 2026-11-24

`tools/bench/batch_gap_bench.cpp --dist Exponential,LogNormal,GammaCDF`, three
runs, each its own process. Why: `PLAN.md`, "Pool wake-up and batch spacing",
measurements 4 and 5. L7 on this machine (Poisson and Gaussian PDF):
`../2026-11-23-kabylake-batch-gap/`.

The three rows span the saving of PARALLEL over VECTORIZED per element: about
1.3–1.9 ns (Exponential PDF), 3–4.5 ns (LogNormal PDF) and 200 ns (Gamma CDF).

## Conditions

- MacBook Pro 14,1, i7-7820HQ (4 cores / 8 threads), macOS 13.7.8, AppleClang
  15, `dev/v3.0.0-corvus-adoption` at `164cbce` (with #324), Release
  `libstats.a`, the bench built with the library's own flags from the working
  tree: `164cbce`'s source plus the `--dist`, `--gaps` and `--sizes` options.
  `--wide-gap` was added to the source after this binary was built.
- Screen locked, AC power. Each run started once the 1-minute load was below
  2.0 (1.78, 1.77, 1.96; `run.txt`).
- Sizes 256 to 200,000. **Forced PARALLEL below 4096 elements times the same
  as VECTORIZED** (ratio 0.93–1.04 in every cell but three: 0.81, 1.12, 1.19): the library does not
  split a batch that small. The tables below start at 4096.

| File | Options |
|---|---|
| `gap2_plain.csv` | only the thread pool |
| `gap2_init.csv` | `--init`: the polling work-stealing pool alive |
| `gap2_plain_sleep.csv` | `--sleep --gaps 0,2000,20000`: the caller sleeps in the gap |

## Result

Smallest measured size from which PARALLEL wins at every larger size:

| Gap | Exponential PDF | LogNormal PDF | Gamma CDF |
|---|---|---|---|
| plain, 0 and 50 µs | 50,000 | 10,000 | 4096 |
| plain, 500 µs to 20 ms | 50,000 (ratio 0.82–0.98 there) | 25,000 | 4096 |
| `--init`, every gap | 50,000 | 10,000 | 4096 |
| plain, caller asleep 2 ms or 20 ms | 50,000 | 25,000 | 4096 |

Extra time per forced-PARALLEL call against back to back, µs, plain, caller
busy, at 4096 / 10,000 / 25,000 / 50,000 / 100,000 / 200,000:

| Gap | Exponential | LogNormal |
|---|---|---|
| 500 µs | 8 / 30 / 6 / 39 / 48 / 49 | 47 / 33 / 34 / 47 / 49 / 34 |
| 2 ms | 59 / 56 / 34 / 63 / 69 / 87 | 63 / 58 / 66 / 81 / 82 / 68 |
| 20 ms | 59 / 56 / 47 / 67 / 71 / 111 | 69 / 63 / 68 / 82 / 107 / 76 |

1. **A gap of 2 ms or more costs a forced-PARALLEL call about what it cost
   on Poisson and Gaussian** (L7's fit: 61–89 µs): mostly 56–82 µs here, 34
   to 111 over all sizes, whatever the kernel costs. For Gamma CDF the same reading is 45
   and 47 µs at 4096 and lost in the run-to-run spread of calls that take 1 to
   17 ms.
2. **That delay predicts where the crossover goes.** PARALLEL wins when the
   saving per call exceeds the delay:
   - Exponential at 50,000 saves 65 µs back to back against a delay of 63–67:
     a tie, and the measured ratio there is 0.97–0.98. At 25,000 it loses
     either way.
   - LogNormal at 10,000 saves 9 µs against 58–63 and loses (ratio 1.5–1.6);
     at 25,000 it saves 81 µs against 66–68 and wins narrowly (0.94–0.96).
     The crossover moves from 10,000 to 25,000.
   - Gamma CDF at 4096 saves 830 µs; the delay is 5% of it and the crossover
     does not move. Delay over saving puts the crossover near 400 elements,
     below the size at which the library splits a batch.
3. **With the polling pool alive the crossovers do not move** and the extra
   time per call at 2 ms and 20 ms is 15 µs or less up to 100,000 elements
   (11–30 at 200,000). Back
   to back it costs VECTORIZED 10% (2.90 against 2.63 ns for Exponential, 6.6
   against 5.9 for LogNormal) and forced PARALLEL at 4096 48% and 29% (8.0
   against 5.4 ns, 9.5 against 7.3).
4. **VECTORIZED pays 6–12 µs per call after a busy gap of 2 ms or more** at
   4096 and 10,000 elements, on both cheap rows, with and without the polling
   pool (L7 saw 10 µs on Gaussian). At 4096 an Exponential call takes 1.8
   times as long (4.8 against 2.6 ns per element).
5. **After a sleeping caller everything is slow, VECTORIZED included**: 3.8×
   for Exponential and 3.7× for LogNormal at 4096 (10.1 against 2.6 ns, 22.0
   against 6.0), 2.2× for Gamma CDF, falling to 2.4–2.9× and 1.0–1.1× at
   200,000. Forced PARALLEL is 5–7× slower at 4096 and 2.4–3.6× at 200,000 on
   the cheap rows. The crossovers land where the busy gap put them, because
   both sides lose the clock.

## Not explained

- The 6–12 µs VECTORIZED step (finding 4). The 256-bit control proposed in
  `PLAN.md`, measurement 6, is what `--wide-gap` is for.
- Gamma CDF forced PARALLEL at 10,000 is slower per element than at 4096 or
  25,000 at every gap (133 against 100 and 92 ns).

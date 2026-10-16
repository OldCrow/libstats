# Zen 4: MSVC against clang-cl, and the overnight clock (2026-10-05)

Indicative, not quiet: screen on, user present, no other load. Freeze head `6cab6ed`
(docs head `af4ed05`).

## Files

- `msvc.csv`, `clangcl.csv`: `tools/bench/v242_cost_bench.cpp` built against `build/`
  (MSVC 19.51, `/arch:AVX512` globally) and `build-clangcl/` (clang-cl 22.1.3, Ninja,
  `-mavx512f` globally), run back to back with `LIBSTATS_BENCH_WARMUP_SECONDS=20`.
- `msvc_vec_report.txt`: `/Qvec-report:2` for the new density loops.
- `clock_probe.cpp`: a dependent scalar chain timed per 5 s.

## MSVC does not vectorize the log1pmx_series loops

Reason 1106 (an outer loop containing an inner loop) at `beta.cpp:309` (Beta Stirling),
`gamma.cpp:315` (Gamma Stirling) and `simd_dispatch.cpp:277` (`vector_expm1`'s Taylor
loop). The inner loop is the Horner loop over a constant array in
`detail::log1pmx_series`. Clang unrolls it and vectorizes the outer loop.

ns per element, n = 1e4, forced VECTORIZED:

| Row | MSVC | clang-cl | M1 (`2026-10-05-m1-v242-cost`) |
|---|---|---|---|
| Beta(25, 25) pdf | 27.5 | 9.4 | 15.1 |
| Beta(1000, 1000) pdf | 26.3 | 8.0 | 13.3 |
| Gamma(20) logpdf | 16.0 | 4.9 | 9.0 |
| Gamma(1000) pdf | 14.4 | 3.6 | 7.1 |
| Beta(2, 2) pdf (older path) | 3.3 | 3.2 | 6.4 |

Over all 960 rows, MSVC/clang-cl geomean 1.25. Scalar-only rows (Student-t CDF and quantile,
von Mises quantile) differ by 1.0–1.17×.

## The overnight R3/R8 runs ran near base clock

The MSVC `v242.csv` from the overnight R3 (`2026-10-04-zen4-v242-cost`) is slower than
`msvc.csv`, the same binary: geomean 1.53× scalar, 1.49× vectorized, 1.53× per call, 1.28×
parallel. That is base clock (3.2 GHz) against boost (~4.7 GHz). PARALLEL lost less than
VECTORIZED, so the R8 crossovers captured that night lean towards PARALLEL.

`clock_probe` with the screen on: 127–133 ms per pass for 90 s (no drop), the same whether
started directly, hidden, or in a minimized window.

### Cause: power throttling under a locked screen

- `lock_default_probe.txt`: screen locked 15 s in; the pass time went 129.5 → 190 ms
  within 5 s (base clock) and stayed there after the unlock until the process ended. A
  new process started after the unlock ran at 129 ms.
- `lock_ab_default_vs_optout.txt`: two probes side by side, one opted out of power
  throttling with `nothrottle.ps1` (`SetProcessInformation`, ProcessPowerThrottling,
  EXECUTION_SPEED). Locked 145 s in: the default probe went to 190 ms, the opted-out one
  stayed at 128 ms.
- The opt-out is per process; a child of an opted-out process reads ControlMask 0.

## The unroll (998d985)

`log1pmx_series`' Horner written out from 1/35 in the same order. MSVC vectorizes the Beta
and Gamma Stirling loops (zmm instructions in the functions 26 → 251 and 18 → 129;
packed FP 12 → 186 and 12 → 98). clang-cl's code was vectorized before and after; it loses
one FMA per polynomial (the step from 0.0: −2 packed, −4 scalar in Beta) and changes
register allocation, nothing else. Neither compiler emits an FMA for MSVC's build
(MSVC: 0 FMAs, no contraction). The accuracy sweep's result bits are identical before and
after under both compilers (10210 rows).

`clangcl_ab_{old,unrolled}_{1,2,3}.csv`: interleaved A/B of clang-cl old and unrolled
(screen on): new/old 0.98 over all rows (medians of three), 0.99 on the large-shape Beta
and Gamma rows. Same-binary run-to-run spread: median 1.07×, p90 1.28×, max 2.55×, so a
single unquiet run cannot resolve 20%.

`vector_expm1`'s loop was tried with the Horner written out, both select operands
precomputed, and an arithmetic blend: MSVC reports reason 1100 for each (the |x| < ½
select), so it was left as it is.

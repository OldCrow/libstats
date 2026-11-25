# Kaby Lake — corvus per call against span length, for #156 (measurement B), 2026-11-24

`tools/bench/corvus_scaling_bench.cpp` at `d23d353` (with the `erfinv` and
`erfcinv` columns of `4db8257`), built against this branch's corvus v1.2.0,
three passes. Why: `PLAN.md`, "#156 — what each machine still owes". Z's run:
`../2026-11-24-zen4-v300-msvc-and-scaling/`.

## Conditions

- MacBook Pro 14,1, i7-7820HQ (AVX2+FMA), macOS 13.7.8, AppleClang 15,
  `-O3 -DNDEBUG -std=c++20`, `libcorvus.a` from the Release build of
  `dev/v3.0.0-corvus-adoption` at `d23d353`. **corvus selects AVX2** (4 lanes).
- Screen locked, AC power. Each pass started below a 1-minute load of 2.0
  (1.99, 1.87, 1.85; `run.txt`). No warm-up option set.
- The directory carries the local date; the run is 01:39–01:44 UTC on
  2026-11-25.

## Result

ns per element, min of 9 runs in each pass, median of the three passes. The
largest max/min over the passes of any cell is 1.12.

| n | erf | exp | lgamma | gamma_p | beta_p | gamma_p_inv | erfinv | erfcinv |
|---|---|---|---|---|---|---|---|---|
| 1 | 38.0 | 43.5 | 119.1 | 655.9 | 2841.1 | 6815.4 | 157.6 | 166.0 |
| 2 | 18.1 | 19.5 | 103.8 | 642.7 | 1528.6 | 6419.4 | 105.4 | 103.7 |
| 4 | 8.4 | 9.1 | 51.2 | 145.6 | 759.5 | 3554.9 | 50.5 | 51.4 |
| 8 | 7.3 | 8.2 | 46.5 | 315.6 | 1020.5 | 2611.2 | 49.5 | 50.3 |
| 32 | 6.5 | 7.4 | 56.5 | 274.1 | 855.9 | 2946.8 | 49.1 | 49.9 |
| 256 | 6.3 | 7.3 | 54.2 | 233.2 | 906.5 | 2921.9 | 46.8 | 48.1 |
| 4096 | 6.3 | 7.3 | 53.8 | 234.2 | 950.4 | 2971.9 | 45.7 | 46.3 |
| 65536 | 6.3 | 7.3 | 53.6 | 234.5 | 936.8 | 2977.8 | 46.2 | 46.8 |

1. **A one-element call costs about a whole 4-lane vector** for `erf`,
   `exp`, `gamma_p` and `beta_p`: a call of four costs 0.84–1.07 times a call
   of one. For `erfinv` and `erfcinv` it costs 0.8 of a vector, for `lgamma`
   0.6 and for `gamma_p_inv` 0.5 (the inputs differ per element and the
   iterative kernels' cost follows them).
2. **Against the per-element rate at long spans, a one-element call costs
   2.2–3.5 elements** for the six costly kernels (`lgamma` 2.2, `gamma_p_inv`
   2.3, `gamma_p` 2.8, `beta_p` 3.0, `erfinv` 3.4, `erfcinv` 3.5) and 6.0 for
   `erf` and `exp`, where the call itself is most of the time. Z, 8 lanes:
   half a vector to a whole one.
3. **The library's scalar paths cost what one such call costs.** From the
   sustained bench on this machine
   (`../2026-11-22-kabylake-v300-cost/sustained/`, pass 2): Gamma scalar CDF
   709 ns against `gamma_p` at 656; Beta scalar CDF 2752 against `beta_p` at
   2841; Gamma quantile 8685 against `gamma_p_inv` at 6815; Gaussian quantile
   174 against `erfcinv` at 166.
4. **What a scalar entry point at the per-element rate could recover**, on
   the same figures: Gamma scalar CDF from 709 to about 290 ns (`main`: 161),
   Beta scalar CDF from 2752 to about 940 (`main`: 153), Gamma quantile from
   8685 to about 3000 at best (`main`: 805). So up to 2.4–2.9×, and Beta's
   CDF and Gamma's quantile would still be 6× and 3.7× above `main`. That
   is an upper bound: it assumes a one-lane kernel as fast per element as the
   4-lane one.

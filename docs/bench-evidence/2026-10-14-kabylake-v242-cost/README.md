# K at d1adc95: R3 costs against v2.4.1, 2026-10-14

Kaby Lake i7-7820HQ, AppleClang 15, macOS 13.7.8, machine locked. Code head `d1adc95`
against `v2.4.1` (`692fd08`, built in a detached worktree). `tools/bench/v242_cost_bench.cpp`
was built per its header against each tree's `build-release/libstats.a`. Each step started
once the 1-min load was below 2.0 (`run_k_final.txt`). The same chain ran R1 and R9
(`../2026-10-14-kabylake-r9-tsan/`).

- `v241.csv`, `v242.csv`: back-to-back runs. `compare.txt`: `tools/bench/v242_compare.py`.
- `timing.txt`: `ctest -j1 -L timing`, 14/14. That is v2.5.0's timing count
  (`docs/test_architecture.md`); the 2026-10-04 run listed 22.

Against K's previous R3 (`../2026-10-04-kabylake-v242-cost/`), the per-case worst and
best are stable, with these exceptions:
- Student-t at ν = 7 and 30: the pdf slowdown of 4.2× is gone.
- Gamma(20) logpdf (VECTORIZED): 26.1 → 29.6 ns, the second-order term of `0ab1b25`.
- Student-t(1000) cdf: 182 → 197 ns (beyond |t| = √ν, `cb99464`).

Section 3 has 49 NEW rows where AUTO is more than 1.25× slower than the best forced
strategy. All are rows where PARALLEL now wins from 1e4: Binomial, NegativeBinomial,
von Mises, Gamma(20/1000) and the Student-t pdf at large ν. They are the R8 finding
(`data/profiles/dispatcher/2026-10-13T*_sha-d1adc95`) seen from the cost side, and they
wait for the table update.

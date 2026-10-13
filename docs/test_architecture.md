# Test Architecture

This document describes the v2.x test organisation.

## Test categories

| Category | Label | Notes |
|---|---|---|
| Correctness | none | Safe for parallel CI runs |
| Timing-sensitive | `timing` | Run serially on a quiet machine |
| Benchmark | `benchmark` | Not part of normal correctness validation |

At v2.5.0, `ctest -N` lists 148 tests: 132 correctness, 14 `timing` and 2
`benchmark`. The `run_tests_correctness` target runs 122 of the 132: it also
excludes tests whose names match `stress|comprehensive|dynamic|_fix|parallel_compilation`.

## Standard correctness command

```bash
ctest --test-dir build --output-on-failure -LE "timing|benchmark"
```

## Split timing cases

A binary that mixes correctness and timing cases is registered twice
(`libstats_split_timing_cases` in `tests/CMakeLists.txt`, #212): `<name>`
runs everything except its timing cases, unlabelled, and `<name>_timing`
runs only those, labelled `timing`. A case is timing when it asserts on a
duration, speedup or ratio of times; printing one does not count. Labelling
a whole binary `timing` would keep its correctness cases out of CI's
correctness run. `<name>_timing_split_names` checks that every named case
still exists.

## Gate and accuracy tests

- `test_*_gates.cpp`: contract and regression gates (`test_contract_gates`,
  `test_batch_nan_gates`, `test_support_boundary_gates`,
  `test_concurrency_gates`, `test_simd_dispatch_gates`, `test_trig_ulp_gates`,
  `test_log_special_gates`; `test_parallel_batch_gates` is `timing`).
- Accuracy gates against high-precision references: checked-in vectors from
  `scripts/gen_*_vectors.py` (mpmath at 40 digits; the von Mises, Gaussian
  and LogNormal CDFs and the SIMD trig/exp/log kernels), and tests with
  mpmath reference values in the source (`test_gamma_density_accuracy`,
  `test_gamma_quantile_accuracy`, `test_incomplete_gamma_beta_accuracy`,
  `test_probit_accuracy`, `test_tail_and_tie_accuracy`,
  `test_survival_tails`, `test_student_t_tails`, `test_von_mises_tails`
  and others).
- `test_distribution_identities` and `test_io_fit_fuzz`: oracle-free
  identities and round trips over all 27 distributions, and stream/fit
  fuzzing.

A new regression guard must be shown to fail against the unfixed state, on
the platform it targets, before it is trusted; see `AGENTS.md`, "Testing".

## Stress test

`test_concurrency_stress` races every distribution's readers against writer
programs (setters, fit, stream, assignment), runs two-writer rounds and a
strategy differential, and asserts values and progress only, so it is a
correctness test. A 120 s per-race watchdog turns a deadlock into a failure.
It runs serially (`RUN_SERIAL`), as does `test_concurrency_gates`. The
environment variable `LIBSTATS_STRESS_SCALE` (default 1) multiplies its
rounds and loop time; CI's ThreadSanitizer leg sets 0.1.

## Analysis tests

Generic analysis coverage:

- `test_goodness_of_fit.cpp`
- generic `stats::analysis` functions in enhanced distribution tests

Distribution-specific analysis coverage:

- `test_gaussian_analysis.cpp`
- `test_poisson_analysis.cpp`
- `test_discrete_analysis.cpp`
- `test_distribution_analysis.cpp` (exponential, gamma, binomial)

## Copy/move tests

Concurrent copy/move stress-testing lives in the TOOL
`tools/copy_move_stress.cpp` (it replaced the old `test_copy_move_stress.cpp`),
covering all 27 distributions under multi-threaded copy/move load. The
`noexcept`-move static assertions
(`std::is_nothrow_move_constructible_v<D>` / `_assignable_v<D>`) live in
`tests/test_copy_move_fix.cpp`.

## SIMD tests

`simd_verification` is the primary SIMD correctness and performance validation tool:

```bash
./build/tools/simd_verification
```

CI can validate SIMD compilation, but real-machine SIMD runtime validation is still needed for release confidence.

## Adding tests

Use GTest for new correctness tests unless a small standalone executable is more appropriate. Register new GTest files with `create_libstats_gtest` in `tests/CMakeLists.txt`.

Keep timing assertions out of correctness tests. If a test depends on wall-clock speed, label it `timing`.

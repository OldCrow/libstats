# Test Architecture

This document describes the v2.x test organisation.

## Test categories

| Category | Label | Notes |
|---|---|---|
| Correctness | none | Safe for parallel CI runs |
| Timing-sensitive | `timing` | Run serially on a quiet machine |
| Benchmark | `benchmark` | Not part of normal correctness validation |

## Standard correctness command

```bash
ctest --test-dir build --output-on-failure -LE "timing|benchmark"
```

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

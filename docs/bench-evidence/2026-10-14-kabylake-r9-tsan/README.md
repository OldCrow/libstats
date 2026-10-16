# K at d1adc95: R1 (Release) and R9 (ThreadSanitizer), 2026-10-14

Kaby Lake i7-7820HQ (4C/8T; the only machine that runs WORK_STEALING), AppleClang 15,
macOS 13.7.8. Code head `d1adc95`; tree clean at `df78931`. The chain script and its
log are in `../2026-10-14-kabylake-v242-cost/` (`run_k_final.sh`, `run_k_final.txt`).

## R1

A fresh `build-release` (release preset, Ninja): 0 warnings. `system_inspector`: AVX2.
Correctness (`-LE "timing|benchmark"`): 132/132 (`r1_ctest.txt`).

## R9

A fresh `build-tsan`: RelWithDebInfo, `-fsanitize=thread` on compile and link, tools
off. 0 warnings, and 889 `__tsan_` symbols in `libstats.a`. Run with
`LIBSTATS_STRESS_SCALE=0.1` and `TSAN_OPTIONS="halt_on_error=0 second_deadlock_stack=1"`.

| Binary | Result | TSan reports |
|---|---|---|
| test_concurrency_stress | 136/136, no skips (#180 fixed: 12–27 skipped before) | 0 |
| test_concurrency_gates | 7/7 | 0 |
| test_snapshot_consistency | 8/8 | 0 |
| test_thread_pool | 13/13 | 0 |
| test_work_stealing_pool | 12/12 | 0 |
| test_parallel_execution_integration | 4/4 | 0 |
| test_parallel_exception_propagation | 14/14 | 0 |
| whole correctness suite (`-j2`) | 132/132 | 0 (TSan exits 66 on a report, which fails the test) |

The chain's first per-binary pass used `build-tsan/tests/<name>` and exited 127 (not
found). Under this configuration the binaries are in `build-tsan/tests/RelWithDebInfo/`.
The table and the `r9_test_*.txt` files are the rerun from that path, on the same
build, the same day.

#183, the positive control in R9's step 4, has been fixed since `b0ac2ad`, so a clean
run is now the expected result. This is the first K TSan run since fix-7's
`readConsistent` (`0ab1b25`) and #180's fix (`8be51a3`..`5213b42`).

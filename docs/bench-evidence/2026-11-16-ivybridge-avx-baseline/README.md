# Ivy Bridge AVX baseline on Linux at 321bb98 (2026-11-16)

First native run of libstats on the fourth fleet machine, and the pre-swap
baseline for the corvus adoption (v3.0.0). `main` at `321bb98`; `src/` and
`include/` there are identical to the `v2.5.0` tag. Correctness and accuracy
only: the machine was in normal use (load average 2–4), so nothing here is a
timing record. Machine and toolchain: `environment.txt`.

## Builds

Fresh Ninja Release configures of the same commit, one per compiler:

- `gcc13/` — GCC 13.3: `cmake -B build-ivb-main -G Ninja -DCMAKE_BUILD_TYPE=Release`.
  0 warnings.
- `clang18/` — Clang 18.1.3 (libstdc++): the same with
  `-DCMAKE_C_COMPILER=clang-18 -DCMAKE_CXX_COMPILER=clang++-18`. 1 warning:
  `kBudgetTight` unused in `tests/test_trig_ulp_gates.cpp` (no FMA tier is
  compiled on this CPU).

Under both, `tools/cpp20_features_inspector` fails to link against TBB (#290),
so the builds ran with `-- -k 0`; every other target built. Configure selects
SSE2 and AVX (the AVX2 and AVX-512 runtime probes fail with an illegal
instruction, as they should), and `system_inspector --quick` confirms the AVX
tier, dispatch gates 3/3.

## Results (identical under both compilers unless stated)

| File | What | Result |
|---|---|---|
| `ctest.txt` | `ctest -LE "timing\|benchmark" -j4` | 131/132; the failure is `WorkStealingPool.ParallelForDoesNotDeadlockOnTaskException` (#288). The log also shows `getOptimalThreadCount()` = 16 on 8 logical cores (#289). |
| `test_trig_ulp_gates.txt` | cos/sin ULP gate | 14/14. AVX cos max 1 ULP, mean 0.0278; sin max 1 ULP, mean 0.0248. |
| `simd_verification.txt` | `tools/simd_verification` | 140 pass, 0 fail. |
| (not stored) | `tools/accuracy_sweep` | 26,882 rows, banner `commit=321bb98 isa=AVX date=2026-11-16`. |
| `oracle.txt` | `tools/accuracy_vs_mpmath.py` (mpmath 1.4.1, dps 50) | 190 self-checks passed, 0 contract violations, 0 oracle fallbacks. About 24 minutes on one core, 3.1 GB peak. |

Neither sweep CSV is stored: at 2.6 MB each they exceed the repo's 1000 KB
file limit, and the sweep is deterministic, so `accuracy_sweep` at this commit
on this machine regenerates them. SHA-256 of the GCC sweep:
`961e27f4b73d8de60d4a2943b690a9e7a72b11e22a67800102b9ee993f58387f`.
The Clang sweep is bit-identical to the GCC one except the four rows in `clang18/sweep_diff_vs_gcc13.txt` (scalar Beta `pdf`/`logpdf` at
x = NaN; the NaN's sign bit differs). The oracle's generated block from the
Clang sweep is identical to the one from the GCC sweep, which is the
`isa=AVX` block in `docs/ACCURACY_CHARACTERIZATION.md`.

## Not run

The 14 timing tests, the batch speed gates and every bench. They need a quiet
machine.

# Ivy Bridge — v2.5.1 fleet legs P1–P4 at 51e1390 (2026-11-23)

The I machine's legs for the v2.5.1 patch release (`PLAN.md`, "v2.5.1 patch
release — fleet legs"). `main` at `51e1390`, whose code is identical to the
validated commit `7069adc` (`git diff --stat 7069adc 51e1390` over `src`,
`include`, `tests`, `tools`, `cmake`, `CMakeLists.txt` and `CMakePresets.json`
is empty).

MacBook Pro 9,1, i7-3820QM (4 cores / 8 threads; AVX, no AVX2, no FMA), Linux
Mint 22.3, kernel 7.0.0-43, glibc 2.39, CMake 4.4.4, Ninja. Governor
`schedutil`, driver `intel_cpufreq` (`timing_environment.txt`).

## Builds

Fresh Ninja Release configures of the same commit in new directories, one per
compiler:

- `gcc13/` — GCC 13.3.0: `cmake -S . -B <dir> -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++`.
  0 warnings.
- `clang18/` — Clang 18.1.3 (libstdc++): the same with
  `-DCMAKE_CXX_COMPILER=clang++-18`. 0 warnings (the baseline's one,
  `kBudgetTight`, went with `9452c6d`).

Both builds complete, `tools/cpp20_features_inspector` included. At the
baseline that tool failed to link against TBB under both compilers (#290,
still open). Why it links now was not investigated.

## Results

| Leg | GCC 13.3 | Clang 18.1.3 | File |
|---|---|---|---|
| P1 tier | AVX | AVX | `system_inspector.txt` |
| P1 cores read | 8 logical, 4 physical | 8 logical, 4 physical | `system_inspector.txt` |
| P1 correctness (`-LE "timing\|benchmark" -j8`) | 132/132 | 132/132 | `ctest.txt` |
| P2 `simd_verification` | 140 pass, 0 fail | 140 pass, 0 fail | `simd_verification.txt` |
| P3 timing (`-j1 -L timing`), quiet | 14/14 | 14/14 | `ctest_timing.txt` |
| P4 accuracy sweep | 26,882 rows, 190 self-checks, 0 contract violations, 0 fallbacks | not run | `gcc13/oracle.txt` |

Against the 2026-11-16 baseline on this machine
(`../2026-11-16-ivybridge-avx-baseline/`, `main` at `321bb98`):

- Correctness was 131/132 under both compilers; the failure was #288
  (`WorkStealingPool.ParallelForDoesNotDeadlockOnTaskException`), fixed in
  `237cb9e`.
- `getOptimalThreadCount()` read 16 (#289); the machine now reads 8 logical
  and 4 physical cores.
- The three gates new since v2.5.0 ran and passed under both compilers:
  `ThreadPoolIntegration.OptimalThreadCountMakesNoSystemCallPerCall`,
  `ContractGates.Issue268_ExponentialSetParametersIsDefined`,
  `ContractGates.Issue316_CauchyQuantileFiniteAtSubnormalP`.

## P3 — timing

The first timing run on `main` on this machine. Run by the user from a
terminal with nothing else open (`timing_run.sh.txt`); each build's run
started once the 1-minute load was below 0.30 (0.29 both times,
`timing_run.txt`). About 20 s per build, 19 s of it
`test_parallel_batch_gates`. `timing_environment.txt` shows `dirty=1`: that is
the regenerated accuracy block (below) in the worktree's document, not a code
change.

## P4 — accuracy sweep and the `isa=AVX` block

GCC build. Scored with `tools/accuracy_vs_mpmath.py` under mpmath 1.4.1.
The sweep banner reads `commit=51e1390  isa=AVX  date=2026-11-23`.

`accuracy_block_AVX.diff` is the whole change to
`docs/ACCURACY_CHARACTERIZATION.md`: the banner line and seven rows of the
`cauchy` section. **The regenerated block is not committed**; whether the
blocks are regenerated for v2.5.1 is decided once every machine's diff is in.
Applying the diff to the document reproduces the block, as long as the
`isa=AVX` block has not changed in between.

| Cauchy row | At `321bb98` | At `51e1390` |
|---|---|---|
| quantile, max abs | 5.403e+289 | 1.505e+289 |
| quantile, max rel | 2.543e-16 | 2.543e-16 |
| cdf, max rel (scalar and batch) | 2.847e-16 | 2.904e-16 |
| pdf, max rel (scalar and batch) | 4.213e-16 | 3.465e-16 |
| logpdf, max abs (scalar and batch) | 2.336e-13 | 2.333e-13 |

The quantile row is #326's change. The cdf, pdf and logpdf rows move although
those functions did not change: the sweep builds their x grid from the
quantile, so the points they are evaluated at moved by an ulp or two. Expect
the same rows to move on the other machines.

**The sweep stamps the commit of the directory it is run from**, not of the
tree its binary was built from. The first sweep here was started from a
folder inside another checkout and its banner named that checkout's commit.
It was rerun from inside the validated checkout: the banner is right and all
26,882 rows are bit-identical to the first run. Only the second run was
scored into the diff here.

## Not done

- No Clang sweep. At the baseline the Clang sweep matched GCC's bit for bit
  except the NaN sign on four Beta rows.
- No dispatch-table capture, cost bench or separate concurrency leg: out of
  the patch's scope (`PLAN.md`).

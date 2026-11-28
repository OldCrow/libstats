# tools/bench — v2.4.1 vs v2.5.0 throughput comparatives

Standalone, not part of the CMake build: each program is compiled twice
against two checkouts (the v2.4.1 tag and the corvus branch) and run back to
back, so the ratio between the two binaries survives a loaded machine even
when the absolute numbers are INDICATIVE. First results (M1 NEON, 2026-09-29)
are in `PLAN.md` Next Steps (c) and issue #156; the Zen 4 record (2026-09-30,
two Windows configurations, warmed) is
`docs/bench-evidence/2026-09-30-zen4-quiet-warm/`, with the two earlier
passes, the runner logs and the build scripts beside it; the Kaby Lake
record (2026-09-30, AVX2, quiet, one pass) is
`docs/bench-evidence/2026-09-30-kaby-quiet/`.

- `distributions_bench.cpp` — public API: batch pdf/logpdf/cdf at 1e6 under
  auto dispatch, scalar pdf/logpdf/cdf at 1e5, quantile at 1e4; ns/element,
  fastest timed block, ten distributions.
- `elementary_bench.cpp` — `VectorOps::vector_exp/log/cos/sin/erf` at 1e3 and
  1e6 (retired tier kernels vs corvus).
- `corvus_scaling_bench.cpp` — corvus called directly: ns/element vs span
  length 1 … 65536, to separate per-call overhead from per-element cost.
- `batch_gap_bench.cpp` — forced VECTORIZED against forced PARALLEL when
  batches are separated by serial work on the caller (0, 50 µs, 500 µs, 2 ms,
  20 ms), Poisson and Gaussian PDF at 4096 … 200000. The profiler times batches
  back to back; this shows whether its crossovers survive a gap long enough for
  the pool's workers to sleep. One binary, built against the tree under test;
  run it plain, with `--init` and with `--init --auto` (three runs, 3 to 5
  minutes each, quiet machine), because the pool state cannot be undone within
  a process. `--dist`, `--gaps` and `--sizes` choose other rows (Exponential,
  Uniform, LogNormal PDF, Gamma CDF), gaps and sizes; `--wide-gap` and
  `--fma-gap` keep a 256-bit add or FMA in the gap loop. The file's header has
  the detail.

Build (Release trees; `OLD` = a `git worktree` of `v2.4.1` with
`libstats_static` built, `NEW` = the branch checkout with its build dir):

```bash
OLD=/path/to/v2.4.1-worktree; NEW=/path/to/libstats; NB=$NEW/build-release
clang++ -std=c++20 -O2 -DNDEBUG -I$OLD/include -I$OLD/include/libstats -I$OLD/build/generated \
  distributions_bench.cpp $OLD/build/libstats.a -lpthread -o bench_v241
clang++ -std=c++20 -O2 -DNDEBUG -I$NEW/include -I$NEW/include/libstats -I$NB/generated \
  distributions_bench.cpp $NB/libstats.a $NB/_deps/corvus-build/libcorvus.a \
  -L/opt/homebrew/lib -lhwy -lhwy_contrib -lpthread -o bench_v250
./bench_v241; ./bench_v250
```

`elementary_bench.cpp` builds the same way; `corvus_scaling_bench.cpp` needs
only `-I<corvus>/include`, `libcorvus.a` and Highway. On x86 replace the
Highway `-L` with wherever `libhwy` lives (or the fetched
`_deps/highway-build`), and on Windows/MSVC use the equivalent `/I` and `.lib`
paths. Use a Release build of the branch and a quiet machine where possible;
label the run INDICATIVE otherwise.

- `pool_wake_bench.cpp` — the follow-up to `batch_gap_bench`: whether spaced
  batches cost more because the pool's workers start late or because they run
  slowly. It drives the library's pools with a fixed integer workload and
  reports, per call, each worker's start delay and its time per element.
  Runs: plain, `--init`, `--init --ws`; about two minutes each. Read the
  result with `scripts/fit_gap_model.py`, which fits the same two models to
  `batch_gap_bench` CSVs.
Per-cell timing (#320): `distributions_bench`, `elementary_bench` and
`table_ab_bench` time every cell through `sustained_timing.h`: the cell's own
work runs for 0.15 s and is discarded, then it is timed in blocks for at least
0.25 s. A cell timed for a few calls reads the CPU clock the previous cell left
behind (50–100 ms to recover after a change of threading mode on the Ivy Bridge
machine), and the fixed cell order makes that bias repeat instead of averaging
out. `LIBSTATS_BENCH_CELL_WARMUP_SECONDS` and `LIBSTATS_BENCH_CELL_SECONDS`
change the two durations. This is separate from the process-level
`LIBSTATS_BENCH_WARMUP_SECONDS` below, which covers a step that arrives seconds
into a run.

## Windows (Zen 4)

An MSVC build of corvus stops at AVX2 (Highway blocklists AVX-512 under
`cl.exe`), and one FetchContent tree cannot mix compilers. To measure corvus at
AVX-512 behind an MSVC libstats, build corvus as a system package with
clang-cl, which keeps the MSVC ABI:

1. From a `vcvars64` shell with the VS-bundled clang-cl on `PATH`, build and
   install Highway 1.4.0 and corvus `v1.2.1` (Ninja, Release,
   `-DCMAKE_C_COMPILER=clang-cl -DCMAKE_CXX_COMPILER=clang-cl`) to one prefix;
   Highway takes the flags the CI install-contract leg uses.
2. Configure both libstats trees with Ninja, Release, `cl`; give the branch
   `-DCMAKE_PREFIX_PATH=<prefix>`. It must report `using system corvus 1.2.1`.
3. Compile each bench with the flags and definitions of the library's own TUs
   (`build.ninja`): `/std:c++20 /O2 /Ob2 /DNDEBUG /EHsc /MD /utf-8
   /arch:AVX512 /DNOMINMAX /D_USE_MATH_DEFINES` plus the four
   `/DLIBSTATS_HAS_*=1`. Without `NOMINMAX` the headers do not compile.
   Link `stats_static.lib`, and for the branch `corvus.lib hwy.lib
   hwy_contrib.lib` from the prefix.

`corvus_scaling_bench` prints the tier corvus selected; it must read
`AVX3_ZEN4`, or the run measured the AVX2 cap.

Set `LIBSTATS_BENCH_WARMUP_SECONDS=25` for the record run on this machine: the
CPU steps down ~1.5× some 8–15 s into sustained load, and without an
in-process warm-up the step lands on different rows of the two binaries, so
ratios under ~2× cannot be read. The warm-up puts every row in the sustained
regime, which is the right one for batch numbers and ~1.5× pessimistic for a
single scalar call. Off by default; the other machines do not need it.

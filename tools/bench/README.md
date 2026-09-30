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
  min of 7 runs, ten distributions.
- `elementary_bench.cpp` — `VectorOps::vector_exp/log/cos/sin/erf` at 1e3 and
  1e6 (retired tier kernels vs corvus).
- `corvus_scaling_bench.cpp` — corvus called directly: ns/element vs span
  length 1 … 65536, to separate per-call overhead from per-element cost.

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

## Windows (Zen 4)

An MSVC build of corvus stops at AVX2 (Highway blocklists AVX-512 under
`cl.exe`), and one FetchContent tree cannot mix compilers. To measure corvus at
AVX-512 behind an MSVC libstats, build corvus as a system package with
clang-cl, which keeps the MSVC ABI:

1. From a `vcvars64` shell with the VS-bundled clang-cl on `PATH`, build and
   install Highway 1.4.0 and corvus `v1.0.1` (Ninja, Release,
   `-DCMAKE_C_COMPILER=clang-cl -DCMAKE_CXX_COMPILER=clang-cl`) to one prefix;
   Highway takes the flags the CI install-contract leg uses.
2. Configure both libstats trees with Ninja, Release, `cl`; give the branch
   `-DCMAKE_PREFIX_PATH=<prefix>`. It must report `using system corvus 1.0.1`.
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

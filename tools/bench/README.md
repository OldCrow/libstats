# tools/bench — v2.4.1 vs v2.5.0 throughput comparatives

Standalone, not part of the CMake build: each program is compiled twice
against two checkouts (the v2.4.1 tag and the corvus branch) and run back to
back, so the ratio between the two binaries survives a loaded machine even
when the absolute numbers are INDICATIVE. First results (M1 NEON, 2026-09-29)
are in `PLAN.md` Next Steps (c) and issue #156.

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

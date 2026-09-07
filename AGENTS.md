# AGENTS.md

Project-scoped guidance for AI agents and contributors. This file is loaded on
every turn of every session in this repo, so it carries orientation and routing
only. Depth lives in `docs/`, in the path-scoped rules under `.claude/rules/`,
and in skills — see the reading map below.

## Project Overview

libstats is a **design and teaching library**: a demonstration of how to build
statistical software correctly in modern C++20, with genuine SIMD and parallel
performance. Zero external dependencies.

**Current status**: v2.4.0 released (tagged on `main`) — 27 distributions across
7 families, API additive over v2.1.0. v1.5.3 is the final v1.x release; v2.0.0
introduced breaking changes, and `MIGRATION_GUIDE.md` has the old→new call
mapping.

Commit-level history is in `CHANGELOG.md` (git-cliff). Per-version validation
matrices and SIMD speedup benchmarks are in `docs/VALIDATION_HISTORY.md`. This
file covers current-state guidance only.

## Session Start

Follow the standard architecture check:
<https://github.com/OldCrow/standards/blob/main/SESSION-START.md>.

Then, because the active SIMD tier changes fundamentally between machines and
code paths, thresholds and test results all depend on it:

- Run `./build/tools/system_inspector --quick` (`.\build\tools\system_inspector.exe --quick`
  on Windows) to confirm the active SIMD capabilities before interpreting any
  performance or test result.
- If the machine changed since the last session, say so explicitly, and
  reconfigure — the build directory may not be current for this architecture.
- Benchmark results are **not** comparable across architectures, and dispatch
  thresholds in `include/libstats/core/dispatch_thresholds.h` are
  architecture-specific.

The tier-to-CPU mapping and which `simd_*.cpp` files compile per tier are in
`docs/SIMD_OPTIMIZATION_REFERENCE.md`.

## Build Commands

```bash
cmake --preset dev                 # default Dev build -> build/
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Other presets: `release` (build-release/), `debug`, `rel-with-debug` (preferred
for profiling), `strict` (warnings as errors, cross-compiler compatibility).
Manual alternative: `cmake -B <dir> -DCMAKE_BUILD_TYPE=<Dev|Release|Debug|RelWithDebInfo|Strict>`.

Build options (`LIBSTATS_VERBOSE_BUILD`, `LIBSTATS_FORCE_TBB`,
`LIBSTATS_BUILD_TOOLS`, `LIBSTATS_BUILD_TESTS`), build-system features, the
build directory map, and the header tooling are documented in
`docs/BUILD_SYSTEM_GUIDE.md` and `docs/HEADER_TOOLS_GUIDE.md`.

Windows uses the Visual Studio x64 Release flow — see Platform-Specific Notes.

## Platform-Specific Notes

| Machine | OS | CPU | SIMD | Role |
|---|---|---|---|---|
| MacBook Pro 14,1 (2017) | macOS Ventura | Intel Kaby Lake | SSE2+AVX+AVX2+FMA | AVX2/FMA validation |
| Mac Mini M1 | macOS Tahoe | Apple Silicon M1 | NEON only | ARM/NEON path validation |
| Asus TUF A16 (2025) | Windows 11 Pro | Ryzen 7 7445 (Zen 4) | +**AVX-512** | Windows/MSVC, only AVX-512 machine |

macOS requires Ventura 13+. On the Zen 4 box, AMD Precision Boost 2 steps down
from ~4.5–5 GHz to a TDP-limited sustained frequency under load. That is a
power constraint, not thermal throttling, and it looks like a dispatch-threshold
anomaly if you don't account for it (`docs/VALIDATION_HISTORY.md`, v2.0.3).

### Windows Session Setup

Toolchain activation, one-time setup, and the Smart App Control/Defender notes:
[Windows Toolchain](https://github.com/OldCrow/standards/blob/main/WINDOWS-TOOLCHAIN.md).
libstats-specific steps after activating the toolchain:

```powershell
Copy-Item "build\Release\stats.dll" -Destination "build\tests\" -Force
ctest --test-dir build -C Release -LE "timing|benchmark" --output-on-failure
```

After any clean rebuild, spot-check `test_gaussian_basic_dynamic.exe` and
`test_exponential_basic_dynamic.exe` with
`dumpbin /imports <exe> | Select-String vcruntime` for a stale Debug CRT
(`VCRUNTIME140D.dll` instead of `VCRUNTIME140.dll`).

## Architecture

Strict **layered dependency architecture**, 6 levels: Foundation (constants,
platform detection) → Core Utilities (math, safety, validation) + Platform
(SIMD, threading) → Advanced Infrastructure (caching, performance framework) →
Distribution Framework (base classes, interfaces) → Concrete Distributions →
Complete Library Interface (`libstats.h`).

Two API surfaces: an **auto-dispatch API** that selects the strategy for you,
and an **explicit strategy API** for direct SIMD/parallel control. Thread safety
rests on lock-free atomic parameter reads on the fast path, a `shared_mutex` for
cache updates, and thread-safe batch operations.

Header layout, include styles, namespace hygiene, and the per-distribution and
analysis header maps are in `docs/HEADER_ARCHITECTURE_GUIDE.md`. The
distribution roster lives in the code and in that guide, not here.

## Batch API contracts

These two are deliberately kept in this file rather than a doc: violating either
produces silently wrong results or a deadlock, not a build error.

- **Input and output spans must not overlap.** Several batch kernels re-read
  `values` after writing `results` (Gaussian/von Mises CDF tail fixups, Gamma
  PDF, LogNormal LogPDF — #112), so an in-place call returns wrong values
  silently. This is a documented contract, not an implementation guarantee: the
  debug-mode `LIBSTATS_ASSERT_NO_OVERLAP` beside the size check in
  `detail::DispatchUtils::autoDispatch`/`executeWithStrategy` compiles away under
  `NDEBUG`, so Release builds do not detect aliasing. Span sizes must match;
  every overload throws otherwise.
- **Exception behaviour differs by strategy.** `ParallelUtils::parallelFor`
  waits for all chunks, then rethrows the first exception in chunk order (#118).
  `WorkStealingPool::parallelFor` deliberately swallows, because its completion
  latch must be decremented on every path or the caller deadlocks. So
  `Strategy::WORK_STEALING` loses what `Strategy::PARALLEL` reports.

Dispatch thresholds are per-(architecture, distribution, operation) in
`dispatch_thresholds.h`, derived from the profiling data in
`data/profiles/dispatcher/`.

## Testing

```bash
ctest --test-dir build -LE "timing|benchmark"   # correctness, parallel-safe
ctest --test-dir build -j1 -L timing            # timing, on a quiet machine
./build/tools/simd_verification                 # SIMD correctness + performance
./build/tools/strategy_profile                  # forced strategies, threshold tuning
./build/tools/system_inspector --performance    # capability analysis
```

Test categories, labels, and the coverage breakdown are in
`docs/test_architecture.md`. Timing tests fail under CPU contention because
parallel strategies show less speedup on a loaded machine — a measurement
problem, not a correctness one.

**A new regression guard must be shown to fail against the unfixed state, on the
platform it targets, before it is trusted.** Two ways a guard can be
structurally unable to fail, both seen on #97:

- **It passes on either side of the bug.** Asserting "Tier 2 is accurate to
  1.6e-7" also passes on a Tier 1 build, so it would never notice a regression.
  Make the assertion two-sided: decide the expected state independently — there,
  from `__cpp_lib_math_special_functions` — and require the library to agree.
- **It never runs.** That guard was first appended to a `timing`-labelled
  binary, and CI's correctness run is `-LE "timing|benchmark"`, so it executed on
  no runner. Green CI meant only that it compiled.

Neither is caught by reading the test; both are caught by running it against the
broken build once. Platform matters too — the defect it guarded (libstdc++
throwing `std::domain_error` from `std::cyl_bessel_i` through a `noexcept` frame)
does not reproduce on MSVC at all. Fleet-wide CI workflow rules — runner budget,
bounded parallelism, ISA hazards on hosted runners, action pinning — are in
[CI House Style](https://github.com/OldCrow/standards/blob/main/CI-HOUSE-STYLE.md)
and `docs/CI_CD_GUIDE.md`.

## Reading map — load on demand, not preemptively

- Header layout, include styles, namespace hygiene → `docs/HEADER_ARCHITECTURE_GUIDE.md`
- Build options, build types, build-system internals → `docs/BUILD_SYSTEM_GUIDE.md`,
  `docs/HEADER_TOOLS_GUIDE.md`
- SIMD tiers, kernel design, threshold miscalibration → `docs/SIMD_OPTIMIZATION_REFERENCE.md`
- Batch/parallel usage and supported operations → `docs/PARALLEL_BATCH_PROCESSING_GUIDE.md`
- Test categories, labels, strategy → `docs/test_architecture.md`
- CI workflows and baseline → `docs/CI_CD_GUIDE.md`
- Accuracy characterization, per-version validation → `docs/ACCURACY_CHARACTERIZATION.md`,
  `docs/VALIDATION_HISTORY.md`
- Adding a distribution → the `add-distribution` skill (invoke it; do not
  reconstruct the checklist from memory)
- Session state, decisions, open questions → `PLAN.md`

<!-- Maintainer note: HTML comments are stripped before this file enters context,
     so they cost no tokens. Use them for provenance and reminders to humans.
     This file was restructured 2026-09-07 from 592 lines; the durable/on-demand
     split is recorded in the dotfiles repo STATUS.md. Keep it under ~150 lines:
     anything that only matters for one part of the tree belongs in
     .claude/rules/ with a paths: filter, and any multi-step procedure belongs
     in a skill. -->

## Rules that load automatically

These are `.claude/rules/*.md` with `paths:` filters — they enter context only
when a matching file is read, so they cost nothing on unrelated turns:

| Rule | Applies when reading |
|---|---|
| `cpp-conventions.md` | `include/**/*.h`, `src`/`tests`/`tools`/`examples` `**/*.cpp` |
| `simd-kernels.md` | `src/simd_*.cpp`, SIMD headers |
| `cmake.md` | `CMakeLists.txt`, `*.cmake`, `CMakePresets.json` |

## Deferred Items

- `vector_floor` + `vector_blend` primitives across all SIMD backends, to enable
  branchless Discrete CDF and Uniform PDF/LogPDF; low priority given existing
  batch-path speedups (Discrete 8–15x, Uniform 39–54x) from amortization
- `vector_lgamma` — too complex, low immediate distribution impact; indefinitely deferred
- SVE (AArch64 beyond NEON) — no hardware in the ecosystem
- SSE4.1 tier — SSE2 magic-number workaround adequate; not worth a dedicated tier

## Open Items

See `PLAN.md` for current status, in-progress work, and open questions.

# AGENTS.md

Project-scoped guidance for AI agents and contributors. This file is loaded on
every turn of every session in this repo, so it carries orientation and routing
only. Depth lives in `docs/` and in skills — see the reading map below.

<!-- Path-scoped rules (.claude/rules/*.md with a paths: filter) were tried here
     on 2026-09-07 and removed the same day. They do work, but only when a file
     is opened with the Read tool: reading the same file with Bash `cat` fires
     nothing, and auto mode steers file access toward Bash. So whether such a
     rule is in context depends on which tool the agent happened to reach for.
     Conventions that must not be missed therefore live in this file. Patterns,
     if you ever do use them, must be repo-relative -- an absolute path matches
     nothing. -->

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

### CMake standard

Full rules: [CMake House Style](https://github.com/OldCrow/standards/blob/main/CMAKE-HOUSE-STYLE.md)
in the fleet standards repo; this section is self-sufficient for this repo. libstats deviations:
- Target-first scoping, `LIBSTATS_`-prefixed options, warnings PRIVATE and
  `PROJECT_IS_TOP_LEVEL`-gated: landed (Phase 3B). Threading detection and
  compiler-flag/warning-set logic live in `cmake/Threading.cmake` and
  `cmake/CompilerFlags.cmake`; tests and tools are registered from their own
  `tests/CMakeLists.txt` and `tools/CMakeLists.txt` via `add_subdirectory`.
  Warnings are applied PRIVATE per-target through `libstats_apply_warnings(target)`
  (defined in `cmake/CompilerFlags.cmake`), called on every object library,
  the final static/shared libs, tests, and tools — GTest is exempt (fetched
  sources never receive our warning flags). Optimization/debug-info flags
  for the custom `Dev`/`Strict` build types come from `CMAKE_CXX_FLAGS_DEV`/
  `CMAKE_CXX_FLAGS_STRICT`, set with the guarded-FORCE idiom
  (`if(NOT var) set(... FORCE)`) — a plain unguarded `set(... CACHE STRING)`
  is a silent no-op here because CMake auto-creates these per-config cache
  entries empty for any custom `CMAKE_BUILD_TYPE` at `project()`, before this
  file is ever included.
- **Grandfathered custom build types**: `Dev` (default) and `Strict`
  (the `-Werror` vehicle) — kept per house-style exception; not to be
  copied into other repos.
- **A configure-time fact that a public header branches on goes in the
  generated `libstats_config.h`, never in `target_compile_definitions`.**
  Template `cmake/libstats_config.h.in`, installed beside the hand-written
  headers, the same mechanism as `libstats_version.h`. Fleet rule:
  [CMake House Style §7](https://github.com/OldCrow/standards/blob/main/CMAKE-HOUSE-STYLE.md#7-install-contract-libhmm-libstats-corvus).
  libstats #97 was the rule's second incident: `$<LINK_ONLY:>` stripped the
  macro from the installed export, so every consumer compiled Tier 2 Bessel
  and an ODR violation against the library's own TUs.
- Install contract conforms: GNUInstallDirs, `libstats-targets` export
  (namespace `libstats::`), kebab `libstats-config.cmake`, `SameMajorVersion`.
- Presets (`CMakePresets.json`, schema 6, min CMake 3.25): `dev` → `build/`
  (default workflow), `release` → `build-release/`, `debug` →
  `build-debug/`, `rel-with-debug` → `build-relwithdebinfo/`, `strict` →
  `build-strict/`. **Deviation from the shared vocabulary**: `release` maps
  to `build-release/` rather than `build/`, because `build/` is already
  claimed by the default `dev` workflow here — grandfathered alongside the
  `Dev` build type.

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

## Coding Conventions

### Code Standards
- **C++20 Required**: Modern features (concepts, spans, execution policies)
- **Header Guards**: Use `#pragma once` (codebase convention)
- **Naming**: CamelCase classes, snake_case functions/variables
- **Memory Management**: Smart pointers, RAII, no raw pointers
- **Error Handling**: Dual API (Result<T> for factories, exceptions for setters)

### Performance Considerations
- Always rebuild after source changes before running tests
- Use `initialize_performance_systems()` for optimal batch performance
- SIMD kernels impose no alignment requirement on caller data: every load/store of a caller buffer is unaligned (`loadu`/`storeu`); aligned ops are used only on internal `alignas` locals
- Large batch operations (>1000 elements) benefit significantly from parallel execution

### Platform-Specific Conventions
- **macOS**: System AppleClang is the default and only supported v2.x compiler path (Ventura 13+).
- **Build artifacts**: Always in `build/tools/` and `build/tests/`, never `bin/`
- **Threading**: GCD preferred on macOS, TBB/OpenMP on Linux/Windows

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

## SIMD kernel conventions

- **A SIMD kernel must never re-read its input array after the corresponding
  store.** Decide every edge fixup from already-loaded registers. This binds
  the `VectorOps` kernel layer, where in-place calls are legal
  (`LogSpaceOps::logSumExpArrayFallback` calls `vector_exp` with
  `a == result`); the distribution batch span overloads are the opposite case
  and promise no in-place safety at all (#112). In-place legality is a
  `VectorOps` property; it stops at that layer.
- **Accuracy claims hold only for tiers validated on native silicon.**
  `LIBSTATS_MAX_SIMD_TIER` (cmake/SIMDDetection.cmake) caps the highest
  compiled x86 tier so lower tiers can run natively on capable hardware; the
  first-ever native SSE2 run is what exposed #74, invisible under Rosetta for
  years.
- **Gather-vs-polynomial transcendentals are settled** (#33,
  `docs/SIMD_BENCHMARK_RESULTS.md`): x86 hardware gather is too expensive to
  beat a polynomial; NEON is the opposite, where a table lookup is nearly
  free. Table kernels are a NEON technique here, not an x86 one.

### FP-contraction rule

**A new error-free transform (Kahan/Neumaier, TwoSum/Fast2Sum, `fma(a,b,-a*b)`
residuals) must spell every intended fusion explicitly, or scope
`-ffp-contract=off` to that file.** No `-ffp-contract` flag is set anywhere in
this build, so every TU takes its compiler default (GCC `fast`, AppleClang
`on`, MSVC/clang-cl off); an unspelled contraction silently breaks the exactness
the transform's proof assumes. The three compensated sequences in
`src/simd_neon.cpp` are safe today (#84, audited 2026-08-16) because every
fusion is an explicit `vfmaq_f64`/`vfmsq_f64` and the one remaining
multiply-then-add is exact by construction.

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

## Deferred Items

- `vector_floor` + `vector_blend` primitives across all SIMD backends, to enable
  branchless Discrete CDF and Uniform PDF/LogPDF; low priority given existing
  batch-path speedups (Discrete 8–15x, Uniform 39–54x) from amortization
- `vector_lgamma` — too complex, low immediate distribution impact; indefinitely deferred
- SVE (AArch64 beyond NEON) — no hardware in the ecosystem
- SSE4.1 tier — SSE2 magic-number workaround adequate; not worth a dedicated tier

## Open Items

See `PLAN.md` for current status, in-progress work, and open questions.

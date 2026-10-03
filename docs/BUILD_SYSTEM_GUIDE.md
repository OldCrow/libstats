# libstats Build System Guide

This guide describes the v2.x build system for libstats.

## Baseline

libstats v2.x requires C++20 and the following minimum compilers:

| Platform | Minimum compiler | Notes |
|---|---|---|
| macOS | AppleClang 15 | macOS 13 Ventura or newer |
| Linux | GCC 13 or Clang 17 | CI exercises GCC 14 and Clang 17; GCC 13 is the CMake-enforced floor (AVX-512 compile workflow only) |
| Windows | MSVC 19.38, or clang-cl (Clang 17 floor; verified with 22.1) | Visual Studio 2022 17.8 or newer; clang-cl is the full-speed build — see "Windows: cl.exe or clang-cl" |

macOS builds use system AppleClang and Apple libc++. The v2.x build path does not support alternate LLVM toolchain setup.

## Windows: cl.exe or clang-cl

Both compilers are supported and produce ABI-compatible libraries. They differ in what happens to corvus:

- **clang-cl** compiles the fetched corvus for every tier up to AVX-512. This is the full-speed build.
- **cl.exe** stops corvus at AVX2 (Highway blocklists AVX-512 under MSVC) and compiles its kernels 3–19× slower. Results are bit-identical. Configure warns when this happens. libstats' own kernels are unaffected.

From a `vcvars64` environment with clang-cl on `PATH` (the Visual Studio "C++ Clang tools" component, or an LLVM install):

```powershell
cmake --preset windows-clang-cl            # Ninja, Release -> build-clangcl/
cmake --build build-clangcl --parallel
ctest --test-dir build-clangcl -LE "timing|benchmark"
```

`windows-clang-cl-strict` is the warnings-as-errors variant. With the Visual Studio generator, select the toolset instead: `cmake -B build -A x64 -T ClangCL`. That generator ignores `CMAKE_CXX_COMPILER` and `CMAKE_BUILD_TYPE`, which is why the presets pin Ninja.

A third arrangement keeps `cl.exe` for libstats and links a corvus and Highway that were built with clang-cl and installed to a prefix (`-DCMAKE_PREFIX_PATH=<prefix>`); `tools/bench/README.md` has the recipe.

clang-cl's driver reads `-Wall` as `-Weverything` and rejects some GNU-style options; `cmake/CompilerFlags.cmake` translates the Clang warning lists for it.

## Dependencies

libstats has one library dependency: [corvus](https://github.com/OldCrow/corvus)
(special and elementary functions, MIT), which itself uses Google Highway
(Apache-2.0). `cmake/FindOrFetchCorvus.cmake` resolves it at configure time:

- `find_package(corvus 1.0 CONFIG)` first — a system corvus, which in turn
  needs a system Highway ≥ 1.4 (corvus's own find-or-fetch rule).
- Otherwise `FetchContent`, pinned to a release tag (`v1.0.1`). The pin is an
  accuracy pin as much as an API pin: bump it only with a
  characterization-sweep regeneration (`docs/ACCURACY_CHARACTERIZATION.md`);
  the `corvus-pin-currency` CI canary flags drift.

Which path ran is printed at configure time and sets `LIBSTATS_CORVUS_PROVIDER`
(`system` / `fetched`). It matters for one thing: **`install` is supported only
with a system corvus.** A fetched corvus is a build-tree target outside the
export set, so `install(EXPORT)` cannot express `libstats_static`'s
`$<LINK_ONLY:corvus::corvus>` against it — the same rule corvus applies to
Highway. The FetchContent path never installs (it is what pylibstats wheels
use), so it never trips this. To install libstats, install Highway and corvus
first (source builds, `-DCMAKE_INSTALL_PREFIX=<prefix>`, then configure
libstats with `-DCMAKE_PREFIX_PATH=<prefix>`); the CI install-contract leg
does exactly that.

corvus dispatches on its own CPUID. `LIBSTATS_MAX_SIMD_TIER` (below) caps
libstats' arithmetic kernels only; the transcendentals and special functions
run on the tier corvus selects.

## Quick start

```bash
cmake -B build
cmake --build build --parallel
ctest --test-dir build --output-on-failure -LE "timing|benchmark"
```

## Build types

| Build type | Purpose |
|---|---|
| `Dev` | Default developer build with light optimisation and debug info |
| `Debug` | Full debug build |
| `Release` | Optimised production build |
| `RelWithDebInfo` | Optimised build with debug symbols |
| `Strict` | Warnings-as-errors compatibility build |

Use `Strict` for warning audits:

```bash
cmake -B build-strict -DCMAKE_BUILD_TYPE=Strict
cmake --build build-strict --parallel
```

## CMake options

```bash
# Verbose configure messages
cmake -B build -DLIBSTATS_VERBOSE_BUILD=ON

# Force TBB even on platforms with native threading support
cmake -B build -DLIBSTATS_FORCE_TBB=ON

# Enable runtime CPU checks when cross-compiling
cmake -B build -DLIBSTATS_ENABLE_RUNTIME_CHECKS=ON

# Disable tools or tests
cmake -B build -DLIBSTATS_BUILD_TOOLS=OFF -DLIBSTATS_BUILD_TESTS=OFF

# Cap the highest compiled x86 SIMD tier (SSE2|AVX|AVX2|AVX512; empty = no cap).
# Runtime dispatch normally picks the highest tier the CPU supports, so lower-tier
# kernels never execute on capable hardware; a cap compiles the higher tiers OUT,
# letting a lower tier run — and be validated or profiled — natively. Used for the
# first native SSE2 run (exposed #74) and the measured kAvx table (2026-09-04
# capped leg). Assert the active tier afterwards: system_inspector --quick, and
# check the archive has no higher-tier vector_* kernel symbols. Since v2.5.0 the
# cap covers libstats' arithmetic kernels only; exp/log/erf/cos/sin and the
# special functions are corvus kernels on corvus's own dispatch.
cmake -B build-avx-cap -DCMAKE_BUILD_TYPE=Release -DLIBSTATS_MAX_SIMD_TIER=AVX
```

## Target layout

The build uses object libraries to preserve layering and improve incremental builds:

1. `libstats_foundation_obj`
2. `libstats_core_utilities_obj`
3. `libstats_platform_obj`
4. `libstats_infrastructure_obj`
5. `libstats_framework_obj`
6. `libstats_distributions_obj`
7. `libstats_simd_obj`

Final targets:

- `libstats_static`
- `libstats_shared`
- `libstats_headers`
- `libstats_simd_interface`

Aliases:

- `libstats::static`
- `libstats::shared`
- `libstats::headers`
- `libstats::simd`

## Include layout

The source tree mirrors the install tree directly: headers live under
`include/libstats/`, so `#include "libstats/core/foo.h"` resolves identically
in the build tree and after `cmake --install` — no shim, symlink, or copy
step is involved (issue #83 removed the previous include-shim machinery,
which cost a configure-time symlink on macOS/Linux and a flat copy plus an
ALL-target refresh on Windows).

The build tree carries the same dual include contract as the installed
package: `<src>/include` (for `#include "libstats/core/foo.h"`),
`<src>/include/libstats` (for the bare `#include "libstats.h"`), and
`<build>/generated` (for the configure-time-generated `libstats_version.h`).

## SIMD detection

SIMD detection lives in `cmake/SIMDDetection.cmake`.

The detector identifies available compile-time backends and adds source files for:

- fallback scalar dispatch
- SSE2
- AVX
- AVX2+FMA
- AVX-512
- NEON

Per-source SIMD flags use `COMPILE_OPTIONS`, not the deprecated `COMPILE_FLAGS` property.

Runtime dispatch still checks CPU capabilities before selecting SIMD paths.

## Threading detection

Threading detection lives in `cmake/Threading.cmake` (`detect_threading_systems()` plus `detect_tbb_unified()`) and sets cache variables for:

- OpenMP
- POSIX threads
- Grand Central Dispatch
- Windows Thread Pool API
- Win32 threads
- TBB

macOS prefers Grand Central Dispatch unless `LIBSTATS_FORCE_TBB=ON` is set.

## macOS deployment target

The build validates `CMAKE_OSX_DEPLOYMENT_TARGET` against the library minimum (13.0 / Ventura) but does not force it:

- If `-DCMAKE_OSX_DEPLOYMENT_TARGET=<version>` is passed and `<version>` is below 13.0, configuration fails with a fatal error.
- If it is not set, the compiler default is used and configuration prints a reminder of the minimum. This avoids the "object file was built for newer macOS version than being linked" linker warnings that occur when the forced target mismatches the system or dependency libraries (e.g. GTest via vcpkg/Homebrew).

To pin explicitly:

```bash
cmake -B build -DCMAKE_OSX_DEPLOYMENT_TARGET=13.0
```

## macOS shared library signing

The shared library target is ad-hoc signed when `codesign` is available. This satisfies macOS Library Validation for locally built libraries.

## Tests

Correctness tests:

```bash
ctest --test-dir build --output-on-failure -LE "timing|benchmark"
```

Timing tests:

```bash
ctest --test-dir build --output-on-failure -j1 -L timing
```

## Tools

Built tools live in `build/tools/`:

```bash
./build/tools/system_inspector --quick
./build/tools/simd_verification
./build/tools/strategy_profile
```

## Troubleshooting

### Header not found

For direct ad hoc compilation outside CMake, add both source include roots
(the dual bare/`libstats/`-prefixed contract, see AGENTS.md):

```bash
-I./include -I./include/libstats
```

### SIMD source does not compile

Run configuration with verbose output:

```bash
cmake -B build -DLIBSTATS_VERBOSE_BUILD=ON
```

Check the SIMD detection summary.

### Timing tests fail under load

Run only correctness tests for normal validation. Timing tests should run serially on an idle machine.

### Windows SDK reserved identifiers in tests

Several identifiers are reserved by Windows SDK headers and must not be used as variable or function names in test or source files:

| Name | Source | Expands to |
|---|---|---|
| `near` | `windef.h` | empty macro |
| `far` | `windef.h` | empty macro |
| `small` | `rpcndr.h` | `typedef char small` |
| `interface` | `objbase.h` | `struct` |

Using these as identifiers causes parse errors on MSVC even though the code compiles cleanly on macOS/Linux. Use unambiguous alternatives (e.g. `tiny` instead of `small`, `within_tol` instead of `near`).

## Where the build logic lives

Moved here from AGENTS.md on 2026-09-07: navigation detail needed when
editing the build, not in every session. The guarded-FORCE gotcha stayed in
AGENTS.md, because it is a silent-failure trap rather than a signpost.

  compiler-flag/warning-set logic live in `cmake/Threading.cmake` and
  `cmake/CompilerFlags.cmake`; tests and tools are registered from their own
  `tests/CMakeLists.txt` and `tools/CMakeLists.txt` via `add_subdirectory`.
  Warnings are applied PRIVATE per-target through `libstats_apply_warnings(target)`
  (defined in `cmake/CompilerFlags.cmake`), called on every object library,
  the final static/shared libs, tests, and tools — GTest is exempt (fetched
  sources never receive our warning flags). Optimization/debug-info flags
  for the custom `Dev`/`Strict` build types come from `CMAKE_CXX_FLAGS_DEV`/

## The #97 config-header incident

Moved here from AGENTS.md on 2026-09-07. The rule it produced — a
configure-time fact a public header branches on goes in the generated
`libstats_config.h`, never in `target_compile_definitions` — stays in
AGENTS.md next to the fleet rule it implements (CMake House Style §7).
This is what happened when it was broken.

  libstats #97 was the rule's second incident: `$<LINK_ONLY:>` stripped the
  macro from the installed export, so every consumer compiled Tier 2 Bessel
  and an ODR violation against the library's own TUs.

v2.5.0 retired the Bessel tiers and with them the generated
`libstats_config.h`; the rule stands for the next configure-time fact a public
header needs.

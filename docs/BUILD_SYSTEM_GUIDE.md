# libstats Build System Guide

This guide describes the v2.x build system for libstats.

## Baseline

libstats v2.x requires C++20 and the following minimum compilers:

| Platform | Minimum compiler | Notes |
|---|---|---|
| macOS | AppleClang 15 | macOS 13 Ventura or newer |
| Linux | GCC 13 or Clang 17 | GCC 14 also validated in CI |
| Windows | MSVC 19.38 | Visual Studio 2022 17.8 or newer |

macOS builds use system AppleClang and Apple libc++. The v2.x build path does not support alternate LLVM toolchain setup.

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
# check the archive has no higher-tier vector_* kernel symbols.
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

Threading detection is unified in one CMake function and sets cache variables for:

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

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

## Include shim

The build tree exposes headers under:

```text
build/include_shim/libstats/
```

This matches the install-tree path (`include/libstats/`) so `#include "libstats/core/foo.h"` works identically in both contexts.

Implementation is platform-guarded:
- **macOS/Linux**: `build/include_shim/libstats` is a directory symlink to `include/`. Header edits are immediately visible to the compiler with no re-run of cmake required.
- **Windows**: a flat copy is used (symlinks require Developer Mode or elevated privileges). A `libstats_refresh_shim` build target re-copies the directory on every `cmake --build` so mid-session edits are picked up automatically.

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

Use the build-tree shim include path:

```bash
-Ibuild/include_shim
```

or project source include path for direct ad hoc compilation:

```bash
-I./include
```

### SIMD source does not compile

Run configuration with verbose output:

```bash
cmake -B build -DLIBSTATS_VERBOSE_BUILD=ON
```

Check the SIMD detection summary.

### Timing tests fail under load

Run only correctness tests for normal validation. Timing tests should run serially on an idle machine.

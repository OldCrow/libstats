# libstats CI/CD Guide

This guide explains the libstats CI/CD setup, how to reproduce failures locally, and how to debug common workflow issues.

The guide is written for contributors and users of the v2.x series. It avoids project-owner-specific setup requirements except where the release process uses additional real hardware validation to explain CI limits.

## CI purpose

CI answers these questions:

- Does the code build on supported compilers?
- Do correctness tests pass across supported operating systems?
- Do headers and packaging paths work for consumers?
- Do formatting, static analysis, and coverage checks catch regressions?

CI does not replace release validation on real SIMD hardware. GitHub-hosted runners cannot reliably provide AVX2+FMA, NEON, and AVX-512 runtime coverage, so release validation supplements CI with real machines.

## Supported CI baseline

The v2.x baseline is:

| Platform | Compiler | Build | Purpose |
|---|---|---|---|
| Ubuntu latest | GCC 13 | Debug | Minimum Linux compiler baseline |
| Ubuntu latest | GCC 14 | Release | Current stable GCC |
| Ubuntu latest | Clang 17 | Release | Minimum Clang baseline |
| macOS 15 | AppleClang 15+ | Release | macOS Ventura+ baseline |
| Windows latest | MSVC 19.38+ | Debug | Debug CRT and DLL validation |
| Windows latest | MSVC 19.38+ | Release | Production Windows validation |

The project requires C++20. Do not reintroduce matrix entries below the baseline unless a maintenance branch explicitly needs them.

## Workflows

### Main CI workflow

The main workflow runs on pull requests and pushes to supported branches. It builds the library, runs the correctness suite, and executes code-quality checks.

Correctness tests should exclude timing-sensitive labels:

```bash
ctest --test-dir build --output-on-failure -LE "timing|benchmark"
```

### AVX-512 compilation workflow

The AVX-512 workflow checks that AVX-512 sources compile with toolchains that support AVX-512 flags. GitHub-hosted runners do not guarantee AVX-512 hardware, so this workflow is a compile-time validation, not a runtime performance validation.

Keep AVX-512 workflow triggers narrow:

- `main`
- `release/*`
- pull requests that touch SIMD sources or CMake SIMD detection

## Local reproduction

### Configure and build

```bash
cmake -B build
cmake --build build --parallel
```

### Correctness suite

```bash
ctest --test-dir build --output-on-failure -LE "timing|benchmark"
```

### Timing-sensitive tests

Timing tests should run serially on a quiet machine:

```bash
ctest --test-dir build --output-on-failure -j1 -L timing
```

### Strict warnings

v2.x uses one compiler-agnostic strict build type:

```bash
cmake -B build-strict -DCMAKE_BUILD_TYPE=Strict
cmake --build build-strict --parallel
```

## macOS notes

macOS builds use system AppleClang and Apple libc++. Alternate LLVM toolchain setup is not part of the v2.x build path.

Homebrew may still be used to locate developer tools such as GoogleTest. This is dependency discovery only; it is unrelated to compiler or standard-library selection.

GTest discovery order:

1. `find_package(GTest)`
2. Homebrew GTest probe on macOS
3. FetchContent fallback

If GTest discovery fails:

```bash
cmake -B build -DLIBSTATS_VERBOSE_BUILD=ON
```

Then inspect the GTest discovery messages.

## Windows notes

Windows dynamic tests require `stats.dll` beside the test executable. The CMake test helper copies the DLL after building dynamic test targets.

If a dynamic test crashes after a clean rebuild, verify the executable is not a stale Debug build linked against the Release DLL:

```powershell
dumpbin /imports build\tests\test_gaussian_basic_dynamic.exe | Select-String vcruntime
```

Expected Release result:

- `VCRUNTIME140.dll`

Bad stale Debug result:

- `VCRUNTIME140D.dll`

Fix:

```powershell
Remove-Item build\tests\test_gaussian_basic_dynamic.exe, build\tests\test_exponential_basic_dynamic.exe -Force
cmake --build build --config Release --target test_gaussian_basic_dynamic test_exponential_basic_dynamic
```

## GitHub CLI debugging

```bash
gh run list --limit 10
gh run view <run-id>
gh run watch <run-id>
gh run download <run-id>
```

For pull request checks:

```bash
gh pr checks <number>
```

## Common failures

### Compiler below baseline

Symptoms:

- `<concepts>` or `<ranges>` failures
- missing C++20 language features
- unexpected `Strict` build failures

Fix: use AppleClang 15+, GCC 13+, Clang 17+, or MSVC 19.38+.

### SIMD compile flag failure

Symptoms:

- AVX2 or AVX-512 source fails to compile
- errors mention unsupported target attributes or missing intrinsics

Check:

```bash
cmake -B build -DLIBSTATS_VERBOSE_BUILD=ON
```

Then inspect the SIMD detection summary.

### Timing failure under load

Correctness tests should not depend on wall-clock speed. Timing tests are labelled `timing` or `benchmark` and should run serially:

```bash
ctest --test-dir build -LE "timing|benchmark" --output-on-failure
ctest --test-dir build -L timing -j1 --output-on-failure
```

### macOS code-signing failure

The shared library is ad-hoc signed after build when `codesign` is available. If signing fails, verify `codesign` exists and that the build directory is writable:

```bash
which codesign
```

## Release validation practice

Project releases supplement CI with real hardware validation because SIMD runtime behaviour depends on CPU features that CI cannot guarantee.

The current project validation set covers:

- Intel AVX2+FMA
- Apple Silicon NEON
- AMD Zen 4 AVX-512

Contributors do not need this exact setup to use or contribute to libstats. CI remains the required correctness gate for normal pull requests. The maintainer records real-hardware validation results in release notes and project guidance when preparing a release.

## CI maintenance checklist

When editing CI:

- Keep the compiler matrix aligned with the documented v2.x baseline.
- Keep correctness tests separate from timing tests.
- Do not reintroduce legacy compiler-specific strict build-type aliases.
- Do not add alternate LLVM toolchain setup paths.
- Keep AVX-512 compilation validation separate from runtime AVX-512 performance validation.
- Prefer target-scoped CMake settings over global flags.

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

The v2.x build matrix (ci.yml) is four legs, all Release — Debug legs were
deliberately dropped:

| Platform | Compiler | Build | Purpose |
|---|---|---|---|
| Ubuntu latest | GCC 14 | Release | Current stable GCC |
| Ubuntu latest | Clang 17 | Release | Minimum Clang baseline |
| macOS 15 | AppleClang | Release | macOS baseline |
| Windows latest | MSVC 19.38+ | Release | Production Windows validation |

GCC 13 is the CMake-enforced minimum. It is not in the main matrix; three
places build with it: the `coverage` job in `ci.yml`, the
`avx512-compilation.yml` matrix, and `avx512-testing.yml`. The project
requires C++20. Do
not reintroduce matrix entries below the baseline unless a maintenance
branch explicitly needs them.

## Workflows

### Main CI workflow (`ci.yml`)

Runs on pull requests to `main` and pushes to `main`, `develop`,
`release/*` and `v*` (doc-only changes are skipped via
`paths-ignore: ['**.md', 'docs/**', 'LICENSE']` — note that on
pull_request events the filter applies to the PR's full changed-file set),
plus a monthly scheduled canary that catches runner-image/toolchain drift.
Pushes to `dev/*` branches do not trigger it; run it on such a branch by
manual dispatch (`gh workflow run ci.yml --ref <branch>`). The AVX-512
workflows have no manual dispatch.

Beyond the build matrix and correctness suite it runs: code-quality checks,
Strict warnings (Linux, `-Werror`), Sanitizers (ASan/UBSan), a Ninja
generator configure, an export-stability-across-reconfigure job (the #90
class), and coverage (GCC 13). The gcc-14 matrix leg also installs to a
prefix, builds and runs `consumer_example` against it, and smoke-tests
pkg-config; these are steps in that leg, not a separate job. Two legs were
added in v2.5.0:

- **`fma-contraction`** (Linux, clang-17, `-mfma -ffp-contract=on`, Release,
  correctness suite only). The other x86 legs build the generic sources
  without FMA, so a contracted `a*b + c` was seen only on the macos-15
  runner (AArch64, where AppleClang contracts by default); that is where
  08d48aa's Gamma rate-product correction was counted twice (fixed, and
  this leg added, in ac72e9a). This leg
  reproduces AppleClang-AArch64 and clang-cl-with-FMA contraction on Linux,
  so the defect class fails with a readable log. The rule it enforces is
  the FP-contraction rule in `AGENTS.md`; `src/gamma.cpp` and
  `src/inverse_gamma.cpp` build with contraction off
  (`docs/BUILD_SYSTEM_GUIDE.md`).
- **`thread-sanitizer`** (Linux, clang-17, `-fsanitize=thread`,
  RelWithDebInfo, correctness suite). Any report fails the job
  (`halt_on_error`). `LIBSTATS_STRESS_SCALE=0.1` runs the stress test at a
  tenth of its rounds. `tests/tsan.supp` suppresses one report: libstdc++'s
  `std::cyl_bessel_i` writes glibc's `signgam` through `::lgamma` (#267),
  which libstats never reads. Linux selects `PARALLEL`, not
  `WORK_STEALING`, so the work-stealing lambdas are race-checked only on the
  SMT Macs, locally.

`test_concurrency_stress` runtime on the macos-15 runner varies 97–372 s on
unchanged code (six runs, 00d1597–d7e9dbd), against 68–102 s on Linux and Windows: shared ARM VMs and
the spin-then-yield barriers, not a regression. The headroom is ctest's
1,500 s default timeout and the stress test's 120 s per-race watchdog. If
macOS nears half the limit, lower `LIBSTATS_STRESS_SCALE` for that job.

Correctness tests should exclude timing-sensitive labels:

```bash
ctest --test-dir build --output-on-failure -LE "timing|benchmark"
```

### corvus in CI

Every build leg resolves corvus through `cmake/FindOrFetchCorvus.cmake`. On
the hosted runners no corvus package exists, so the FetchContent path runs —
the path pylibstats wheels use. The one exception is the install-contract leg
(Linux, gcc-14, Release): it builds and installs Highway 1.4.0 and corvus
v1.0.1 into `$GITHUB_WORKSPACE/_install` first, because libstats' `install`
target is supported only with a system corvus (`docs/BUILD_SYSTEM_GUIDE.md`,
"Dependencies"); `consumer_example` and the pkg-config smoke test then run
against that prefix. The `reconfigure-export` guard (#90) installs the same
prefix, since the export exists only with a system corvus and `corvus::corvus`
is itself an imported target of the class that guard watches; it now pins
`corvus::corvus` in the export alongside `Threads::Threads`.
`corvus-pin-currency` (monthly cron / manual dispatch)
checks that the find_package floor and the FetchContent tag agree and that
the pin has not fallen behind corvus's newest release.

### AVX-512 compilation workflow

The AVX-512 workflow checks that AVX-512 sources compile with toolchains that support AVX-512 flags. GitHub-hosted runners do not guarantee AVX-512 hardware, so this workflow is a compile-time validation, not a runtime performance validation.

It triggers on pushes to `main` and `release/*`, PRs to `main` (doc-only
changes excluded via paths-ignore), and a monthly scheduled canary. A
separate runtime workflow (`avx512-testing.yml`) builds AND runs the suite
on the same pushes as `ci.yml` and the monthly canary, and on a pull
request only when it carries the `avx512-test` label; it runs on
`ubuntu-latest` and, when the runner has no `avx512f`, skips the build and
tests with a notice. `docs.yml` builds documentation and
`lint-workflows.yml` lints the workflows themselves (actionlint/zizmor).

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

CI's Windows legs use cl.exe (Visual Studio generator), the correctness
build. clang-cl (Ninja) is the local performance build, used for every
timing and threshold capture; both take the same `/arch:` flag
(`docs/BUILD_SYSTEM_GUIDE.md`, "SIMD detection"). A Windows timing run
outside CI needs every measured process opted out of power throttling
(`docs/VALIDATION_HISTORY.md`, "Zen 4 timing: power throttling").

The Visual Studio generator writes Release test executables to
`build\tests\Release\` (per-config output directories,
`tests/CMakeLists.txt`); single-config Ninja writes them to `build\tests\`.
Windows dynamic tests require `stats.dll` beside the test executable. The CMake test helper copies the DLL after building dynamic test targets.

If a dynamic test crashes after a clean rebuild, verify the executable is not a stale Debug build linked against the Release DLL:

```powershell
dumpbin /imports build\tests\Release\test_gaussian_basic_dynamic.exe | Select-String vcruntime
```

Expected Release result:

- `VCRUNTIME140.dll`

Bad stale Debug result:

- `VCRUNTIME140D.dll`

Fix:

```powershell
Remove-Item build\tests\Release\test_gaussian_basic_dynamic.exe, build\tests\Release\test_exponential_basic_dynamic.exe -Force
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

Correctness tests should not depend on wall-clock speed. Timing tests are labelled `timing` or `benchmark`; run them serially on a quiet machine (see "Timing-sensitive tests" above).

### macOS code-signing failure

The shared library is ad-hoc signed after build (`codesign --force -s -`) when `codesign` is available. `--force` lets the step run again on an already-signed dylib: after a reconfigure, Ninja re-runs the symlink step and the signing without relinking, and before v2.5.0 that failed with "is already signed" (#162). On an older tree, clear it with `codesign --remove-signature <dylib>` and rebuild.

If signing fails otherwise, verify `codesign` exists and that the build directory is writable:

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

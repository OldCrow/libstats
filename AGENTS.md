# AGENTS.md

This file provides project-scoped guidance to AI agents and contributors working in this repository.

## Project Overview

libstats is a **design and teaching library**: a demonstration of how to build statistical software correctly in modern C++20, with genuine SIMD and parallel performance. Zero external dependencies.

**Current status**: v2.1.0 on `main` — 19 distributions across 7 families, 46/46 correctness tests pass on Kaby Lake AVX2+FMA, Mac Mini M1 NEON, and Asus TUF A16 AVX-512 (CI validated; audit-remediation re-validation completed pre-v2.0.4, see `docs/VALIDATION_HISTORY.md`). v1.5.3 is the final v1.x release.

For the full commit-level history, see `CHANGELOG.md` (auto-generated via git-cliff). For historical per-version validation matrices and SIMD speedup benchmarks, see `docs/VALIDATION_HISTORY.md`. This file covers current-state guidance only.

v2.0.0 introduced breaking changes relative to v1.5.3 (final v1.x release).
See `MIGRATION_GUIDE.md` for the complete old→new call mapping.

## Session Start

At the start of every session, perform these steps in order:

1. Verify machine architecture before making SIMD assumptions.
2. Select the matching build path (macOS vs Windows/MSVC, Intel vs Apple Silicon).
3. Reconfigure/rebuild when the machine or architecture differs from the previous session context.

Quick architecture checks:

```bash
# macOS/Linux shells
uname -m
uname -s
sysctl -n machdep.cpu.brand_string 2>/dev/null || true
```

```powershell
# PowerShell (Windows)
[System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture
[System.Runtime.InteropServices.RuntimeInformation]::ProcessArchitecture
$env:PROCESSOR_IDENTIFIER
```

### Why SIMD detection matters

The active SIMD tier changes fundamentally between machines. SIMD code paths, performance thresholds, and test results are architecture-dependent. If the machine has changed since the last session:
- Note the change explicitly.
- Verify the build directory is current for this architecture (`cmake ..` may be needed).
- Dispatch thresholds in `include/core/dispatch_thresholds.h` are architecture-specific.
- Benchmark results are not comparable across architectures.

| SIMD Tier | Example CPUs | Active simd_*.cpp files |
|---|---|---|
| SSE2 + AVX + AVX2 + FMA | Intel Haswell / Kaby Lake and newer | + `simd_avx2.cpp` |
| NEON only | Apple Silicon (M1 and newer) | `simd_neon.cpp` |
| SSE2 + AVX + AVX2 + **AVX-512** | AMD Zen 4 (e.g. Ryzen 7000-series) | + `simd_avx512.cpp` |
| SSE2 + AVX + AVX2 | Linux x86 CI | `simd_sse2.cpp`, `simd_avx.cpp`, `simd_avx2.cpp` |

The machines in the Development Ecosystem table are examples; any CPU with the same SIMD capabilities follows the same code paths.

Platform routing rules (OS/toolchain selection — SIMD tier is determined automatically at compile time by CPU feature detection):
- **macOS (Ventura 13+ required):** Use the standard CMake flow in the Build Commands section.
- **Windows/MSVC:** Follow Platform-Specific Notes below and use Visual Studio 2022 x64 Release commands (defaults shown for Asus TUF A16; paths may differ on other machines).
- **All platforms:** After architecture verification, run `./build/tools/system_inspector --quick` (Unix shells) or `.\build\tools\system_inspector.exe --quick` (Windows PowerShell) to confirm active SIMD capabilities before interpreting performance/test results.

### Current validation matrix (v2.1.0)

| Machine | SIMD | Correctness | Timing | Notes |
|---|---|---|---|---|
| Mac Mini M1 | NEON | 46/46 ✅ | 22/22 ✅ | Validated 2026-07-05 |
| Kaby Lake (2017 MBP) | AVX2+FMA | 46/46 ✅ | — | CI validated |
| Asus TUF A16 (Windows) | AVX-512 | 46/46 ✅ | — | CI validated |

For every prior release's validation matrix and SIMD speedup tables, see `docs/VALIDATION_HISTORY.md`.

## Agent Workflow

- When reviewing repository state or "what's changed" (e.g., syncing after time away, catching up on a branch), start with `git diff --stat` and `git log` rather than reading full file contents. Read complete files only for items you've determined are directly relevant to the task at hand.
- For any subagent expected to run more than ~30 minutes, structure its brief to report interim progress at natural milestones (e.g., after each major deliverable) rather than running silently to a single final report.

## Build Commands

### Quick Build
```bash
# macOS/Linux — standard development build (default 'Dev' build type, output in build/)
cmake --preset dev
cmake --build build --parallel   # equivalent to make -j$(nproc)
ctest --test-dir build --output-on-failure
```

Manual alternative (no preset): `cmake -B build -DCMAKE_BUILD_TYPE=Dev && cmake --build build`.

Windows: use the commands in Platform-Specific Notes below.

### Common Build Configurations
```bash
# Development (default) - light optimization with debug info (build/)
cmake --preset dev

# Production release - maximum optimization (build-release/)
cmake --preset release

# Full debugging support (build-debug/)
cmake --preset debug

# Release with debug symbols — preferred for profiling (build-relwithdebinfo/)
cmake --preset rel-with-debug

# Strict compiler warnings as errors, for compatibility testing (build-strict/)
cmake --preset strict   # v2.0.0: unified Strict mode replaces legacy compiler-specific strict aliases
```

Manual alternative: `cmake -B <dir> -DCMAKE_BUILD_TYPE=<Dev|Release|Debug|RelWithDebInfo|Strict>`.

### CMake Options
```bash
# Enable verbose build messages for debugging
cmake -DLIBSTATS_VERBOSE_BUILD=ON ..

# Force TBB usage over platform-native threading
cmake -DLIBSTATS_FORCE_TBB=ON ..

# Disable tools or tests
cmake -DLIBSTATS_BUILD_TOOLS=OFF -DLIBSTATS_BUILD_TESTS=OFF ..
```

The build system supports cross-compiler compatibility testing with specialized build types that enable consistent warning levels across GCC, Clang, and MSVC.

### CMake standard

Full rules: `CMAKE-HOUSE-STYLE.md` in the Development root on dev machines (master copy, not checked in); this section is self-sufficient for this repo. libstats deviations:
- Target-first scoping, `LIBSTATS_`-prefixed options, warnings PRIVATE and
  `PROJECT_IS_TOP_LEVEL`-gated: in place for the current object-library
  hierarchy; deeper modularization (cmake/ modules for warning sets,
  tests/tools subdirectory CMakeLists) is Phase 3 work, not yet landed.
- **Grandfathered custom build types**: `Dev` (default) and `Strict`
  (the `-Werror` vehicle) — kept per house-style exception; not to be
  copied into other repos.
- Install contract conforms: GNUInstallDirs, `libstats-targets` export
  (namespace `libstats::`), kebab `libstats-config.cmake`, `SameMajorVersion`.
- Presets (`CMakePresets.json`, schema 6, min CMake 3.25): `dev` → `build/`
  (default workflow), `release` → `build-release/`, `debug` →
  `build-debug/`, `rel-with-debug` → `build-relwithdebinfo/`, `strict` →
  `build-strict/`. **Deviation from the shared vocabulary**: `release` maps
  to `build-release/` rather than `build/`, because `build/` is already
  claimed by the default `dev` workflow here — grandfathered alongside the
  `Dev` build type.

### Build System Features
- **Automatic parallel detection**: Detects CPU cores and configures optimal builds
- **Compiler detection**: System AppleClang on macOS (Ventura 13+); GCC 13+ / Clang 17+ on Linux
- **SIMD optimization**: Runtime CPU feature detection with fallbacks
- **Cross-platform**: Native Windows, macOS, Linux support

### Important Build Directories
- **Executables**: `build/tools/` (never `bin/` - this doesn't exist)
- **Tests**: `build/tests/`
- **Examples**: `build/` (built by examples/CMakeLists.txt)

### Development Tools
```bash
# System analysis and diagnostics
./build/tools/system_inspector --full
./build/tools/cpp20_features_inspector

# Performance analysis
./build/tools/strategy_profile
./build/tools/simd_verification

# Dispatcher profiling bundle capture
./scripts/capture_dispatcher_profile.sh

# Cross-compiler compatibility testing
./scripts/test-cross-compiler.sh --clean
```

### Ad Hoc Compilation Outside CMake

For quick diagnostics and testing, compile directly without CMake. Use the system compiler on
macOS (Ventura 13+); alternate LLVM compiler setup is not required and not supported in v2.0.0.

```bash
# macOS — system AppleClang (recommended)
clang++ -std=c++20 -stdlib=libc++ \
  -I./include \
  -L./build \
  your_test.cpp -o test_output ./build/libstats.a

# Linux — GCC 13+ or Clang 17+
g++ -std=c++20 -Wall -Wextra -O2 \
  -I./include \
  -L./build \
  your_test.cpp -o test_output -lstats
```

Quick test template:
```cpp
#include "libstats.h"
#include <iostream>

int main() {
    auto result = stats::GaussianDistribution::create(0.0, 1.0);
    if (result.isOk()) {
        auto& g = *result;  // operator* returns T& (Result<T> redesigned 2026-07-01)
        std::cout << "PDF at 0: " << g.getProbability(0.0) << "\n";
        std::cout << "CDF at 1: " << g.getCumulativeProbability(1.0) << "\n";
    }
}
```

Troubleshooting:
- **Library not found**: Use static linking (`./build/libstats.a`) instead of `-lstats`.
- **Header not found**: Verify `-I./include` path is correct relative to the project root.
- **C++20 features not available**: Ensure compiler version meets minimum (AppleClang 15, GCC 13, Clang 17).

## Platform-Specific Notes

### Development Ecosystem

| Machine | OS | CPU | SIMD | Notes |
|---|---|---|---|---|
| MacBook Pro 14,1 (2017) | macOS Ventura | Intel Kaby Lake | SSE2 + AVX + AVX2 + FMA | AVX2/FMA validation |
| Mac Mini M1 | macOS Tahoe | Apple Silicon M1 | NEON only | ARM/NEON path validation |
| Asus TUF A16 (2025) | Windows 11 Pro | AMD Ryzen 7 7445 (Zen 4) | SSE2 + AVX + AVX2 + **AVX-512** | Windows/MSVC + first AVX-512 machine |

The Asus TUF A16 (Ryzen 7 7445, Zen 4) is the first machine in this ecosystem with AVX-512 support. AMD Precision Boost 2 steps down from boost (~4.5–5 GHz) to TDP-limited sustained frequency under sustained 100% CPU load — this is a thermal-stable power constraint, not thermal throttling, and can look like a dispatch-threshold anomaly if not accounted for (see `docs/VALIDATION_HISTORY.md` v2.0.3 notes).

### Windows Session Setup

> **Windows tool paths vary** by installation method (direct installer, `winget`, `chocolatey`, Microsoft Store, etc.). The paths below are common defaults — adjust for your installation. VS Build Tools and full VS editions use different default directories; see One-time setup notes below for alternatives and auto-detection.

Before building or running tests in a new PowerShell session on Windows:

```powershell
# 1. Activate MSVC toolchain (required each session — not persistent in PowerShell)
# Default path for VS 2022 Build Tools. For full VS (Community/Professional/Enterprise),
# replace "BuildTools" with your edition under "C:\Program Files\Microsoft Visual Studio\2022\".
# See One-time setup notes below for auto-detection via vswhere.exe.
$vcvars = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
$envVars = cmd /c "`"$vcvars`" > nul && set"
foreach ($line in $envVars) {
    if ($line -match "^([^=]+)=(.*)$") {
        [System.Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], 'Process')
    }
}

# 2. Set UTF-8 output (required for Unicode glyphs in tool output)
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

# 3. Ensure stats.dll is accessible for dynamic linking tests
Copy-Item "build\Release\stats.dll" -Destination "build\tests\" -Force

# 4. Run correctness tests
ctest --test-dir build -C Release -LE "timing|benchmark" --output-on-failure
```

**Important: After any clean rebuild on Windows, verify the dynamic test EXEs are Release builds:**
```powershell
dumpbin /imports build\tests\test_gaussian_basic_dynamic.exe | Select-String vcruntime
# Must show VCRUNTIME140.dll (Release), NOT VCRUNTIME140D.dll (Debug)
# If Debug CRT is shown, the EXE is a stale Debug binary. Fix:
#   Remove-Item build\tests\test_gaussian_basic_dynamic.exe, test_exponential_basic_dynamic.exe -Force
#   cmake --build build --config Release --target test_gaussian_basic_dynamic test_exponential_basic_dynamic
```
The VS generator puts Debug and Release test EXEs in the same `build\tests\` directory.
A stale Debug EXE + Release DLL = CRT mismatch = heap corruption crash. The `cmake --build --clean-first`
flag cleans Release artifacts but leaves existing Debug EXEs untouched if their timestamps appear current.

**One-time setup notes:**
- Visual Studio 2022 Build Tools (not full IDE) is sufficient. Install from https://aka.ms/vs/17/release/vs_buildtools.exe, or `winget install Microsoft.VisualStudio.2022.BuildTools`, or `choco install visualstudio2022buildtools`.
  - Build Tools default path: `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\`
  - Full VS (Community/Professional/Enterprise) default path: `C:\Program Files\Microsoft Visual Studio\2022\{edition}\`
  - Auto-detect installation path (any edition): `& "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -property installationPath`
- **Smart App Control must be Off** (Windows Security → App & Browser Control → SAC settings).
  SAC blocks locally compiled executables. Cannot be re-enabled without a Windows reset.
- CMake ≥ 3.25 required. Install from https://cmake.org/download/, `winget install Kitware.CMake`, or `choco install cmake`.
- vcpkg for GTest: `git clone https://github.com/microsoft/vcpkg C:\vcpkg && C:\vcpkg\bootstrap-vcpkg.bat`. The path `C:\vcpkg` is a convention; if installed via `winget install Microsoft.vcpkg` or `choco install vcpkg` the location will differ — run `where vcpkg` to find it.
- Configure: `cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake`
  (adjust the toolchain path if vcpkg is not at `C:\vcpkg`)
- Build: `cmake --build . --config Release --parallel`
- GTest installed via vcpkg (`gtest:x64-windows 1.17.0`) — all 33 correctness tests pass

## Architecture

### High-Level Structure

libstats follows a strict **layered dependency architecture** with 6 levels:

```
Level 0: Foundation (constants, basic platform detection)
Level 1: Core Utilities (math, safety, validation) + Platform (SIMD, threading)
Level 2: Advanced Infrastructure (caching, performance framework)
Level 3: Distribution Framework (base classes, interfaces)
Level 4: Concrete Distributions (Gaussian, Exponential, etc.)
Level 5: Complete Library Interface (libstats.h)
```

### Key Architectural Concepts

#### Dual API Design
- **Auto-dispatch API**: Intelligent automatic strategy selection (recommended for most users)
- **Explicit strategy API**: Direct control over SIMD/parallel execution for power users

#### Performance Systems
- **SIMD Optimization**: Cross-platform runtime detection (SSE2/AVX/AVX2/NEON)
- **Parallel Execution**: Auto-dispatching between scalar, SIMD, and parallel strategies
- **Adaptive Cache**: Performance-aware caching with memory optimization
- **Performance History**: Machine learning for strategy selection improvement

#### Thread Safety
- **Lock-free fast paths**: Atomic parameter access for high-frequency operations
- **Reader-writer locks**: Shared_mutex for cache updates
- **Concurrent batch operations**: Thread-safe SIMD and parallel processing

### Core Components

#### Statistical Distributions (19 implemented, across 7 families)
1. **Gaussian** (Normal) - N(μ, σ²)
2. **Exponential** - Exp(λ)
3. **Uniform** - U(a, b)
4. **Poisson** - P(λ)
5. **Discrete** - Custom discrete distributions
6. **Gamma** - Γ(α, β)
7. **Chi-squared** - χ²(ν) — delegation wrapper over Gamma(α=ν/2, β=1/2)
8. **Student's t** - t(ν) — SIMD log-space PDF/LogPDF and CDF via incomplete beta
9. **Beta** - Beta(α, β) — two-log SIMD PDF/LogPDF and CDF via regularized incomplete beta
10. **Log-Normal** - LogN(μ, σ) — log+exp pipeline
11. **Pareto** - Pareto(xₘ, α) — log-only pipeline, power-law tail
12. **Weibull** - W(k, λ) — log+exp pipeline, reliability engineering
13. **Rayleigh** - R(σ) — x² pipeline, signal processing
14. **Von Mises** - VM(μ, κ) — circular distribution, SIMD via vector_cos
15. **Binomial** - B(n, p) — discrete, PMF via lgamma
16. **Negative Binomial** - NB(r, p) — discrete, real-valued r, Newton–Raphson MLE
17. **Geometric** - Geo(p) — discrete, delegate over NegBinomial(r=1); MLE: p̂=1/(1+x̄)
18. **Laplace** - Laplace(μ, b) — standalone, fabs+vector_exp SIMD; MLE: median/MAD
19. **Cauchy** - Cauchy(x₀, γ) — delegate over StudentT(ν=1); moments NaN; Fisher-scoring MLE

Each implemented distribution provides: PDF/CDF/Quantiles, Statistical Moments, Parameter Estimation (MLE), Random Sampling, Statistical Validation, SIMD batch operations.

#### Platform Optimization
- **CPU Feature Detection**: Runtime SIMD capability detection
- **Threading Systems**: Comprehensive detection (TBB, OpenMP, pthreads, GCD, Windows Thread Pool)
- **Memory Management**: SIMD-aligned allocations and cache-aware algorithms

### Code Organization

Header architecture:
```
include/
├── libstats.h              # Complete library (single include)
├── core/                   # Core mathematical and statistical components
│   ├── constants/          # Mathematical, precision, statistical constants
│   ├── distribution_type.h     # DistributionType enum (append-only)
│   ├── distribution_meta.h     # kDistributionMeta[] — canonical registration table
│   ├── dispatch_thresholds.h   # Per-architecture parallel thresholds (indexed by DistributionType)
│   ├── distribution_*.h    # Distribution framework components
│   └── *_common.h         # Consolidated headers for faster compilation
├── distributions/          # Concrete distributions (gaussian.h, etc.)
├── stats/
│   └── analysis/           # Statistical tests and estimators (stats::analysis::)
│       ├── analysis.h      # Umbrella include
│       ├── goodness_of_fit.h, bootstrap.h, cross_validation.h, information_criteria.h
│       └── gaussian_analysis.h, poisson_analysis.h, exponential_analysis.h, …
└── platform/              # SIMD, threading, parallel execution
```

Source organization:
```
src/
├── [Level 0-1] Foundation and utilities (cpu_detection.cpp, safety.cpp)
├── [Level 2] Platform capabilities (thread_pool.cpp, work_stealing_pool.cpp)
├── [Level 3] Infrastructure (benchmark.cpp, performance_dispatcher.cpp)
├── [Level 4] Framework (distribution_base.cpp)
└── [Level 5] Distributions (gaussian.cpp, exponential.cpp, etc.)
```

Object library architecture: the CMake system uses dependency-aware object libraries for parallel compilation — `libstats_foundation_obj` → `libstats_core_utilities_obj` → `libstats_infrastructure_obj` → `libstats_framework_obj` → `libstats_distributions_obj`. Enables optimal incremental builds and clear architectural boundaries.

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
- SIMD operations require 16-byte aligned data (handled automatically)
- Large batch operations (>1000 elements) benefit significantly from parallel execution

### Platform-Specific Conventions
- **macOS**: System AppleClang is the default and only supported v2.x compiler path (Ventura 13+).
- **Build artifacts**: Always in `build/tools/` and `build/tests/`, never `bin/`
- **Threading**: GCD preferred on macOS, TBB/OpenMP on Linux/Windows

## Common Development Tasks

### Creating New Distributions

The registration checklist is authoritative in `include/core/distribution_meta.h`. Geometric (16), Laplace (17), and Cauchy (18) are the most recently implemented (2026-06-28); for any future distribution (N+1), follow all 6 steps below.

**Steps for any future distribution (N+1):**

1. **Append** the new `DistributionType` enum value to `include/core/distribution_type.h`
   (append-only; never reorder — values are used as array indices).
2. **Append** a `DistributionMeta` row to `kDistributionMeta[]` in `include/core/distribution_meta.h`
   (enum name, display name, `is_discrete`, `is_delegation_wrapper`). Bump the
   `static_assert(kDistributionTypeCount >= N, ...)` minimum to match the new count.
3. **Append** one `ThresholdRow` to each of the four `kXxx` tables in
   `include/core/dispatch_thresholds.h` (use `{NEVER, NEVER, NEVER}` until profiled).
   For delegation wrappers (e.g. Geometric→NegBinomial, Cauchy→StudentT), the delegate's
   thresholds apply — copy them or leave NEVER and profile after implementation.

4. **Implement** the distribution:

   *Header* `include/distributions/dist.h` — use `exponential.h` as the reference:
   - Inherit from `DistributionBase`.
   - Declare `static constexpr detail::DistributionType kDistributionType = detail::DistributionType::DIST_NAME;`
     and `static constexpr bool kIsDiscrete = false/true;` (must match the metadata row).
   - Declare `noexcept` move constructor and move assignment operator.
   - Declare `static void parallelBatchFit(const std::vector<std::vector<double>>&, std::vector<DistType>&);`
   - Override all pure virtuals from `DistributionInterface`: `getMean`, `getVariance`, `getSkewness`,
     `getKurtosis`, `getNumParameters`, `getDistributionName`, `isDiscrete`,
     `getSupportLowerBound`, `getSupportUpperBound`, `getProbability`, `getLogProbability`,
     `getCumulativeProbability`, `getQuantile`, `sample` (×2), `fit`, `reset`, `toString`.
   - Override `getEntropy()` and `getMedian()` (both have NaN defaults in the interface;
     concrete implementations are required even for wrappers).
   - Declare the three batch span overloads: `getProbability(span, span, hint)`,
     `getLogProbability(span, span, hint)`, `getCumulativeProbability(span, span, hint)`.
   - Declare comparison operators (`==`, `!=`) and friend stream operators (`<<`, `>>`).

   *Source* `src/dist.cpp`: full implementations in the numbered section structure.

   *Basic test* `tests/test_dist_basic.cpp`:
   - `#include "include/basic_test_runner.h"`
   - Define `stats::tests::BasicDistConfig cfg{name, small_values, lo, hi, invalid_scenarios};`
   - Keep Tests 1–5 and 7 per-distribution.
   - Call `stats::tests::runBatchTests(cfg, dist);` for Test 6.
   - Call `stats::tests::runErrorTests(cfg);` for Test 8.

   *Enhanced test* `tests/test_dist_enhanced.cpp`:
   - `#include "include/enhanced_test_suite.h"`
   - Implement `template<> struct stats::tests::DistTraits<DistType> : stats::tests::DistTraitsDefaults { ... };`
     with `make()`, `domain()`, `batch_lo()`, `batch_hi()`, `invalid_creators()`.
     Override tolerances for distributions whose SIMD path has documented approximation error
     (e.g. VonMises pdf_tolerance = 1e-10 for vector_cos).
   - Close with `INSTANTIATE_TYPED_TEST_SUITE_P(Name, DistributionEnhancedTest, ::testing::Types<DistType>);`
   - Add per-distribution tests: known analytical values, moment formulas, special cases,
     VectorizedMatchesScalar, VectorizedSpeedup (timing-labelled), MLEFit.

5. **Register** in four CMakeLists.txt locations and in `include/libstats.h`:

   *`CMakeLists.txt` — `LIBSTATS_DISTRIBUTIONS_SOURCES`* (~line 780):
   Add `src/dist.cpp` to the Level-5 distributions source list.

   *`CMakeLists.txt` — test registration* (~line 1380):
   ```cmake
   create_libstats_test(test_dist_basic tests/test_dist_basic.cpp)
   create_libstats_gtest(test_dist_enhanced tests/test_dist_enhanced.cpp)
   ```

   *`CMakeLists.txt` — `run_all_tests` DEPENDS block* (~line 1500):
   Add `test_dist_basic` and `test_dist_enhanced` to the dependency list.

   *`CMakeLists.txt` — timing label* (if the enhanced test has speedup assertions):
   Add `test_dist_enhanced` to the `set_tests_properties(... PROPERTIES LABELS "timing")` call.

   *`include/libstats.h`* — inside `#ifdef LIBSTATS_FULL_INTERFACE`:
   - Add `#include "distributions/dist.h"`
   - Add `using DistName = DistNameDistribution;` in the `namespace stats { ... }` type-alias block.

6. **Profile and calibrate thresholds** (after correctness tests pass on all target machines):
   - Run `./build/tools/strategy_profile --large --export` to produce a CSV.
   - Run `./build/tools/threshold_validator <csv>` to compare measured crossovers against
     the current NEVER entries and identify which need updating.
   - Update the four `kXxx` tables in `dispatch_thresholds.h` accordingly.
   - For delegation wrappers, verify the delegate's thresholds apply (skip if identical).

The `consteval validateMetaOrdering()` in `distribution_meta.h` enforces step 1↔2 alignment at
compile time. A clean build after any enum or table change verifies consistency.

### SIMD Development
- Use `libstats::simd::*` namespace for vectorized operations
- Runtime dispatch automatically selects best available instruction set
- Test with `./build/tools/simd_verification`

### Parallel Processing
- Auto-dispatch API: `getProbability(std::span<const double>, std::span<double>, hint)`
- Explicit control: span-based batch APIs with `detail::PerformanceHint`
- Dispatch thresholds are per-(architecture, distribution, operation) in `dispatch_thresholds.h`
- Thresholds derived from four-architecture profiling data in `data/profiles/dispatcher/`

## CI / Validation

### Running Tests
```bash
# Run all tests (timing assertions may be flaky under parallel load)
ctest --test-dir build --output-on-failure

# Correctness only — safe to run in parallel, excludes timing-sensitive assertions
ctest --test-dir build --output-on-failure -LE "timing|benchmark"

# Timing validation — run serially on a quiet machine for reliable results
ctest --test-dir build --output-on-failure -j1 -L timing

# Or via make targets (macOS/Linux with Makefile generator only)
make run_tests          # Correctness suite (parallel-safe)
make run_tests_timing   # Timing suite (serial, quiet machine required)
make run_all_tests      # Everything
# Windows equivalent: cmake --build build --target run_tests --config Release

# Run a specific test
ctest --test-dir build -R test_gaussian_basic
ctest --test-dir build -R test_gaussian_enhanced  # Contains timing assertions

# Run cross-compiler compatibility tests
./scripts/test-cross-compiler.sh
```

### Test Labels
- **no label** — correctness tests; safe to run in parallel
- **timing** — contains speedup/overhead assertions; run with `-j1` for reliable results
- **benchmark** — performance benchmarks; not part of the standard test suite

Timing tests fail under CPU contention because parallel strategies show less speedup
when the machine is loaded. This is a measurement problem, not a correctness problem.

### Testing Strategy
- **All levels**: GTest-based tests registered with CTest
- Correctness tests: run `ctest -LE "timing|benchmark"` (parallel-safe)
- Timing tests: run `ctest -j1 -L timing` on a quiet machine
- **Coverage**: 50 CTest targets (each basic and enhanced test file registers as one target;
  each enhanced binary runs additional typed test cases from the shared `DistributionEnhancedTest` suite)

### Performance Validation
```bash
# Verify SIMD operations and performance
./build/tools/simd_verification

# Profile forced strategies for threshold tuning
./build/tools/strategy_profile

# System capability analysis
./build/tools/system_inspector --performance
```

## Deferred Items

- `vector_floor` + `vector_blend` primitives across all SIMD backends to enable
  branchless Discrete CDF and Uniform PDF/LogPDF; low priority given existing batch-path speedups
  (Discrete 8–15x, Uniform 39–54x) already achieved through amortization
- `vector_lgamma` — too complex, low immediate distribution impact; indefinitely deferred
- SVE (AArch64 beyond NEON) — no hardware in the ecosystem
- SSE4.1 tier — SSE2 magic-number workaround adequate; not worth a dedicated tier

## Warp Terminal Saved Workflows (warp.dev only)

> **Note for non-Warp users:** These workflows are available only in the Warp terminal. Users of other tools (Claude Code, Cursor, bare shells, etc.) should run the equivalent shell commands listed elsewhere in this file.

Saved workflows in `.warp/workflows/` are available directly in the Warp terminal for common tasks:

- **libstats: Clean Rebuild** — remove `build/` and rebuild from scratch; accepts `build_type` arg (default: `Dev`)
- **libstats: Validate Machine** — architecture detection, SIMD capabilities, correctness suite, and `simd_verification`; requires a current build
- **libstats: Switch Branch + Rebuild** — stash uncommitted changes, fetch, checkout target branch, pull, and clean rebuild in one step
- **libstats: Warning Audit** — build with a strict warning mode and display deduplicated warning counts; accepts `build_type` arg (default: `ClangWarn`)

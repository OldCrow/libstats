# AGENTS.md

This file provides project-scoped guidance to AI agents and contributors working in this repository.

## Project Overview

libstats is a **design and teaching library**: a demonstration of how to build statistical software correctly in modern C++20, with genuine SIMD and parallel performance. Zero external dependencies.

**Current status**: v2.3.0 on `main` — 19 distributions across 7 families, API unchanged from v2.1.0. All three fleet machines validated natively: Asus TUF A16 AVX-512 2026-08-20 (53/53), Kaby Lake AVX2+FMA 2026-08-22 (53/53), Mac Mini M1 NEON 2026-08-23 (55/55 — count-definition note in PLAN.md). See the validation matrix below. v1.5.3 is the final v1.x release.

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
- Dispatch thresholds in `include/libstats/core/dispatch_thresholds.h` are architecture-specific.
- Benchmark results are not comparable across architectures.

| SIMD Tier | Example CPUs | Compiled simd_*.cpp files (runtime dispatch picks the highest supported) |
|---|---|---|
| SSE2 + AVX + AVX2 (+ FMA) | Linux x86 CI; Intel Haswell / Kaby Lake and newer | `simd_sse2.cpp`, `simd_avx.cpp`, `simd_avx2.cpp` |
| SSE2 + AVX + AVX2 + **AVX-512** | AMD Zen 4 (e.g. Ryzen 7000-series) | the three above + `simd_avx512.cpp` |
| NEON only | Apple Silicon (M1 and newer) | `simd_neon.cpp` |

`simd_fallback.cpp` and `simd_dispatch.cpp` are always compiled on every platform (`cmake/SIMDDetection.cmake`).

The machines in the Development Ecosystem table are examples; any CPU with the same SIMD capabilities follows the same code paths.

Platform routing rules (OS/toolchain selection — SIMD tier is determined automatically at compile time by CPU feature detection):
- **macOS (Ventura 13+ required):** Use the standard CMake flow in the Build Commands section.
- **Windows/MSVC:** Follow Platform-Specific Notes below and use the Visual Studio x64 Release commands (VS 2022 17.8+ or later; defaults shown for Asus TUF A16, whose toolchain is now VS 18 (2026) — paths and generator names vary by version and edition, so users creating forks should verify their setup).
- **All platforms:** After architecture verification, run `./build/tools/system_inspector --quick` (Unix shells) or `.\build\tools\system_inspector.exe --quick` (Windows PowerShell) to confirm active SIMD capabilities before interpreting performance/test results.

### Current validation matrix (v2.3.0)

| Machine | SIMD | Correctness | Timing | Notes |
|---|---|---|---|---|
| Asus TUF A16 (Windows) | AVX-512 | 53/53 ✅ | 21/22 ⚠️ | Native, 2026-08-20, MSVC Release |
| Mac Mini M1 | NEON | 55/55 ✅ | 22/22 ⚠️ | Native, 2026-08-23, AppleClang 21 Release, v2.3.0 tag, Bessel Tier 2; timing ran on a loaded machine (indicative only); 55 = `ctest -LE timing` (see PLAN count note) |
| Kaby Lake (2017 MBP) | AVX2+FMA | 53/53 ✅ | 22/22 ✅ | Native, 2026-08-22, AppleClang Release (`release` preset); Bessel Tier 2 (libc++ has no `cyl_bessel_i`) |

The correctness count grew 49 → 53 with v2.3.0's four new accuracy gates:
`test_trig_ulp_gates`, `test_vonmises_cdf_accuracy`, `test_lognormal_cdf_accuracy`,
`test_gaussian_cdf_accuracy`.

The Zen 4 timing failure is the same `UniformEnhancedTest.
SIMDAndParallelBatchImplementations` speedup assertion carried from the
v2.2.0 matrix (1.5x on that run, 1.44x on this one, both against a 1.8x
adaptive threshold at 5000 elements on a settled machine) — flaky, not a
regression;
nothing in v2.3.0 touches uniform kernels or the dispatch thresholds it
measures. Timing tests carry the `timing` label and are excluded from CI
everywhere, so this is a real-hardware finding, not a CI one.
On Kaby Lake the same gate passed twice (Vectorized 22.9x at 5000
elements), both runs serial (`ctest -j1 -L timing`) with one core held by
a stray `exchangesyncd` — a handicap on the parallel path, so the pass is
conservative.

The mpmath accuracy characterization (`docs/ACCURACY_CHARACTERIZATION.md`,
#46) is PROVISIONAL until every ISA block is regenerated in place: the Kaby
Lake sweep is recorded (2026-08-23, delta section), and the M1 NEON sweep ran
2026-08-23 (42/42 self-checks, 76 contract violations vs Zen 4's 86; findings
and deltas in PLAN.md) but its `isa=NEON` block is not yet regenerated into
the doc.

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
./build/tools/threshold_validator <csv>   # compare measured crossovers against dispatch_thresholds.h

# Accuracy characterization vs mpmath (#46; docs/ACCURACY_CHARACTERIZATION.md)
./build/tools/accuracy_sweep              # C++ side: emits the sweep rows
python tools/accuracy_vs_mpmath.py        # Python side: mpmath references + report
# (full registry: tools/CMakeLists.txt — 12 tools plus 2 behind LIBSTATS_BUILD_SIMD_DEV_TOOLS)

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
  -I./include -I./include/libstats \
  -L./build \
  your_test.cpp -o test_output ./build/libstats.a

# Linux — GCC 13+ or Clang 17+
g++ -std=c++20 -Wall -Wextra -O2 \
  -I./include -I./include/libstats \
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
- **Header not found**: Verify `-I./include -I./include/libstats` paths are correct relative to the project root — the bare `#include "libstats.h"` template above resolves via `-I./include/libstats`, while any `#include "libstats/core/foo.h"`-style include resolves via `-I./include`.
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
# Locate the newest installed Visual Studio (any version or edition) with vswhere:
$vsPath = & "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -property installationPath
$vcvars = "$vsPath\VC\Auxiliary\Build\vcvars64.bat"
# Or pin an explicit path, e.g. VS 2022 Build Tools:
#   "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
# or a full edition: "C:\Program Files\Microsoft Visual Studio\{version}\{edition}\VC\Auxiliary\Build\vcvars64.bat"
# ({version} is 2022 for VS 17.x and 18 for VS 2026; {edition} is Community/Professional/Enterprise).
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
- Visual Studio Build Tools (not full IDE) are sufficient; VS 2022 (17.8+) or later. Install from https://visualstudio.microsoft.com/downloads/ (Build Tools for 2022: https://aka.ms/vs/17/release/vs_buildtools.exe, `winget install Microsoft.VisualStudio.2022.BuildTools`, or `choco install visualstudio2022buildtools`).
  - Build Tools default path: `C:\Program Files (x86)\Microsoft Visual Studio\{version}\BuildTools\`
  - Full VS (Community/Professional/Enterprise) default path: `C:\Program Files\Microsoft Visual Studio\{version}\{edition}\` (`{version}` = 2022 for VS 17.x, 18 for VS 2026)
  - Auto-detect installation path (any edition): `& "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -property installationPath`
- **Smart App Control must be Off** (Windows Security → App & Browser Control → SAC settings).
  SAC blocks locally compiled executables. Cannot be re-enabled without a Windows reset.
- CMake ≥ 3.25 required. Install from https://cmake.org/download/, `winget install Kitware.CMake`, or `choco install cmake`.
- GTest needs no manual install: `tests/CMakeLists.txt` tries `find_package(GTest)`, then a Homebrew probe, then a `FetchContent` fallback — the same path CI uses (`cmake -B build ... -A x64`, no toolchain file). A vcpkg-installed GTest is picked up by step 1 if you pass `-DCMAKE_TOOLCHAIN_FILE=<vcpkg>/scripts/buildsystems/vcpkg.cmake`, but it is optional.
- Configure: `cmake .. -A x64` (CMake selects the newest installed Visual Studio; pin one with e.g. `-G "Visual Studio 17 2022"` if several are installed)
- Build: `cmake --build . --config Release --parallel`

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
19. **Cauchy** - Cauchy(x₀, γ) — PDF/LogPDF delegate to StudentT(ν=1), CDF/Quantile closed-form (#48); moments NaN; Fisher-scoring MLE

Each implemented distribution provides: PDF/CDF/Quantiles, Statistical Moments, Parameter Estimation (MLE), Random Sampling, Statistical Validation, SIMD batch operations.

#### Platform Optimization
- **CPU Feature Detection**: Runtime SIMD capability detection
- **Threading Systems**: Comprehensive detection (TBB, OpenMP, pthreads, GCD, Windows Thread Pool)
- **Memory Management**: SIMD-aligned allocations and cache-aware algorithms

### Code Organization

Header architecture:
```
include/libstats/           # Mirrors the installed header layout
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

Object library architecture: the build compiles the sources in seven OBJECT libraries — `libstats_foundation_obj`, `libstats_core_utilities_obj`, `libstats_platform_obj`, `libstats_infrastructure_obj`, `libstats_framework_obj`, `libstats_distributions_obj`, `libstats_simd_obj` — whose objects are combined into the static and shared libraries. They are parallel-compilation groupings, not a dependency chain: there are no `target_link_libraries` edges between them and all seven receive the same include-dir set, so the 6-level layering above is a source-organisation convention, not a build-enforced boundary (an include-layering check script would be the enforceable artefact).

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

### SIMD kernel conventions

- **A SIMD kernel must never re-read its input array after the corresponding
  store.** This binds the `VectorOps` kernel layer, where in-place calls are
  legal (`LogSpaceOps::logSumExpArrayFallback` calls `vector_exp` with
  `a == result`): a post-store re-read sees internally-computed values, not the
  input. Decide every edge fixup from already-loaded registers. It cost a real
  `exp(-inf)` bug during the #33 productionization. Whether the distribution
  batch span overloads promise in-place safety is a separate, currently
  undocumented question (review 2026-08-21) — do not assume it.
- **Accuracy claims hold only for tiers validated on native silicon.**
  `LIBSTATS_MAX_SIMD_TIER` (cmake/SIMDDetection.cmake) caps the highest
  compiled x86 tier so lower tiers can run natively on capable hardware; the
  first-ever native SSE2 run is what exposed #74, invisible under Rosetta for
  years.
- **Gather-vs-polynomial transcendentals are settled** (#33,
  `docs/SIMD_BENCHMARK_RESULTS.md`): x86 hardware gather is too expensive on
  both Kaby Lake (interleave 8.6× an FMA) and Zen 4 (1.70×); NEON is the
  opposite — an Array-of-Structs table pulled by one `vld1q` makes a two-value
  lookup nearly free. Table kernels are a NEON technique here, not an x86 one.
  Do not reopen the x86 half without new hardware.

## Common Development Tasks

### Creating New Distributions

The registration checklist is authoritative in `include/libstats/core/distribution_meta.h`. Geometric (16), Laplace (17), and Cauchy (18) are the most recently implemented (2026-06-28); for any future distribution (N+1), follow all 6 steps below.

**Steps for any future distribution (N+1):**

1. **Append** the new `DistributionType` enum value to `include/libstats/core/distribution_type.h`
   (append-only; never reorder — values are used as array indices).
2. **Append** a `DistributionMeta` row to `kDistributionMeta[]` in `include/libstats/core/distribution_meta.h`
   (enum name, display name, `is_discrete`, `is_delegation_wrapper`). Bump the
   `static_assert(kDistributionTypeCount >= N, ...)` minimum to match the new count.
3. **Append** one `ThresholdRow` to each of the four `kXxx` tables in
   `include/libstats/core/dispatch_thresholds.h` (use `{NEVER, NEVER, NEVER}` until profiled).
   For delegation wrappers (e.g. Geometric→NegBinomial, Cauchy→StudentT), the delegate's
   thresholds apply — copy them or leave NEVER and profile after implementation.

4. **Implement** the distribution:

   *Header* `include/libstats/distributions/dist.h` — use `exponential.h` as the reference:
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

5. **Register** in four CMakeLists.txt locations (one top-level, three in
   `tests/CMakeLists.txt`) and in `include/libstats/libstats.h`:

   *`CMakeLists.txt` (top-level) — `LIBSTATS_DISTRIBUTIONS_SOURCES`*, in the
   "Level 5: Distribution Implementations" block:
   Add `src/dist.cpp` to the Level-5 distributions source list.

   *`tests/CMakeLists.txt` — Level-5 registration block* (under the
   "LEVEL 5 TESTS: Concrete Distribution Implementations" banner comment):
   ```cmake
   create_libstats_test(test_dist_basic test_dist_basic.cpp)
   create_libstats_gtest(test_dist_enhanced test_dist_enhanced.cpp)
   ```

   *`tests/CMakeLists.txt` — `run_all_tests` DEPENDS block* (the
   `add_custom_target(run_all_tests ...)` near the end of the file):
   Add `test_dist_basic` and `test_dist_enhanced` to the dependency list.

   *`tests/CMakeLists.txt` — timing label* (if the enhanced test has speedup assertions):
   Add `test_dist_enhanced` to the `set_tests_properties(... PROPERTIES LABELS "timing")` call.

   *`include/libstats/libstats.h`* — inside `#ifdef LIBSTATS_FULL_INTERFACE`:
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
- Auto-dispatch API: `getProbability(std::span<const double>, std::span<double>, hint)`. Sizes must match (every overload throws otherwise). **Input and output spans must not overlap**: several batch kernels re-read `values` after writing `results` (Gaussian/von Mises CDF tail fixups, Gamma PDF, LogNormal LogPDF — #112), so an in-place call returns wrong values silently; the contract is "no aliasing" until #112 decides otherwise
- Explicit control: span-based batch APIs with `detail::PerformanceHint`
- Dispatch thresholds are per-(architecture, distribution, operation) in `dispatch_thresholds.h`
- Thresholds derived from four-architecture profiling data in `data/profiles/dispatcher/`

## CI / Validation

Fleet-wide workflow rules (runner budget, bounded parallelism, ISA hazards on
hosted runners, action pinning):
[CI House Style](https://github.com/OldCrow/standards/blob/main/CI-HOUSE-STYLE.md).

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

**A new error-free transform must be contraction-proofed where it is written.**
Kahan/Neumaier summation, TwoSum/Fast2Sum, and `fma(a,b,-a*b)` residual tricks
are exact identities whose proofs assume each IEEE operation rounds as written.
A compiler contraction landing inside one makes the "exact" correction term the
error of an operation that never happened — and nothing fails, so a
single-compiler suite cannot notice. No `-ffp-contract` flag is set anywhere in
this build, so every TU takes its compiler default: GCC `fast`, AppleClang `on`,
MSVC/clang-cl off.

The three compensated sequences in `src/simd_neon.cpp` are safe today (#84,
audited 2026-08-16) and are safe *because of how they are written*, not because
of any build setting: every intended fusion is spelled as an explicit
`vfmaq_f64`/`vfmsq_f64`, so no rounded multiply sits adjacent to an add; and the
one remaining multiply-then-add, log's `e*ln2_hi + L_hi`, has a product that is
exact by construction (42-bit constant × ≤11-bit exponent). Adding a compensated
sequence that does neither would reintroduce the hazard silently. So: spell every
fusion, or arrange for the product to be exact, or scope `-ffp-contract=off` to
that file.

**A new regression guard must be shown to fail against the unfixed state, on the
platform it targets, before it is trusted.** Two ways a guard can be structurally
unable to fail, both seen on #97:

- **It passes on either side of the bug.** Asserting "Tier 2 is accurate to
  1.6e-7" also passes on a Tier 1 build, so it would never notice a regression.
  Make the assertion two-sided — decide the expected state independently (there,
  from `__cpp_lib_math_special_functions`) and require the library to agree.
- **It never runs.** The first version of that guard was appended to a
  `timing`-labelled binary, and CI's correctness run is `-LE "timing|benchmark"`,
  so it executed on no runner. A green CI meant only that it compiled.

Neither is caught by reading the test. Both are caught by running it against the
broken build once — and platform matters, since the defect it guards (libstdc++
throwing `std::domain_error` from `std::cyl_bessel_i` through a `noexcept` frame)
does not reproduce on MSVC at all.

### Testing Strategy
- **All levels**: GTest-based tests registered with CTest
- Correctness tests: run `ctest -LE "timing|benchmark"` (parallel-safe)
- Timing tests: run `ctest -j1 -L timing` on a quiet machine
- **Coverage**: 77 CTest targets — 53 correctness (`ctest -C <cfg> -LE "timing|benchmark"`), 22 timing (`-L timing`), 2 benchmark (each basic and enhanced test file registers as one target;
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
- **libstats: Warning Audit** — build with a strict warning mode and display deduplicated warning counts; accepts `build_type` arg (default: `Strict`; the legacy compiler-specific names are not build types since v2.0.0)

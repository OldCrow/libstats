# AGENTS.md

This file provides project-scoped guidance to AI agents and contributors working in this repository.

# libstats - Modern C++20 Statistical Distributions Library

## Project Overview

libstats is a **design and teaching library**: a demonstration of how to build statistical software correctly in modern C++20, with genuine SIMD and parallel performance. Zero external dependencies.

**Current Status**: v1.3.0 released on `main`.
16 distributions across 6 families, full cross-platform SIMD validation (AVX, AVX2, AVX-512, NEON/MSVC),
54/54 SIMD tests passing on all four target machines, 39/39 correctness tests.
v1.3.0: added `BinomialDistribution` and `NegativeBinomialDistribution` (Tier 3 discrete
distributions); `detail::trigamma` added to `math_utils`; 4 new test files (2 basic,
2 GTest enhanced). v1.2.0: all 11 standalone tests migrated to GTest (#19),
`categorizeBatchSize` complexity reduced (#20), dispatch threshold functions consolidated
(#21), shared `parallelBatchFit` helper extracted and added to all 9 distributions (#18),
two pre-existing bugs fixed (LogSpaceOps lookup table precision, DiscreteDistribution
equal-bounds validation).

## Session Start Baseline Workflow (Required)

At the start of every session, perform these steps in order:

1. Verify machine architecture before making SIMD assumptions.
2. Select the matching build path (macOS vs Windows/MSVC, Intel vs Apple Silicon, Catalina vs non-Catalina).
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

### Why SIMD Detection Matters

The active SIMD tier changes fundamentally between machines. SIMD code paths, performance thresholds, and test results are architecture-dependent. If the machine has changed since the last session:
- Note the change explicitly.
- Verify the build directory is current for this architecture (`cmake ..` may be needed).
- Dispatch thresholds in `include/core/dispatch_thresholds.h` are architecture-specific.
- Benchmark results are not comparable across architectures.

| SIMD Tier | Example CPUs | Active simd_*.cpp files |
|---|---|---|
| SSE2 + AVX | Intel Ivy Bridge and earlier | `simd_sse2.cpp`, `simd_avx.cpp` |
| SSE2 + AVX + AVX2 + FMA | Intel Haswell / Kaby Lake and newer | + `simd_avx2.cpp` |
| NEON only | Apple Silicon (M1 and newer) | `simd_neon.cpp` |
| SSE2 + AVX + AVX2 + **AVX-512** | AMD Zen 4 (e.g. Ryzen 7000-series) | + `simd_avx512.cpp` |
| SSE2 + AVX + AVX2 | Linux x86 CI | `simd_sse2.cpp`, `simd_avx.cpp`, `simd_avx2.cpp` |

The machines in the Development Ecosystem table are examples; any CPU with the same SIMD capabilities follows the same code paths.

Platform routing rules (OS/toolchain selection — SIMD tier is determined automatically at compile time by CPU feature detection):
- **macOS (non-Catalina):** Use the standard CMake flow in the `Essential Build Commands` section.
- **macOS Catalina (10.15):** No separate bootstrap script is required in `libstats`, but keep Catalina caveats in `docs/BUILD_SYSTEM_GUIDE.md` in mind (notably Homebrew LLVM 22 `std::format` behavior and dylib code-signing expectations).
- **Windows/MSVC:** Follow `Windows Session Setup` below and use Visual Studio 2022 x64 Release commands (defaults shown for Asus TUF A16; paths may differ on other machines).
- **All platforms:** After architecture verification, run `./build/tools/system_inspector --quick` (Unix shells) or `.\build\tools\system_inspector.exe --quick` (Windows PowerShell) to confirm active SIMD capabilities before interpreting performance/test results.

### Current Validation Matrix (16 distributions, 39 correctness tests, 54 SIMD tests)

| Machine | SIMD | Correctness | Total suite | simd_verification | Speedup |
|---|---|---|---|---|---|
| Ivy Bridge (2012 MBP) | AVX | 39/39 ✅ | 53 | 54/54 ✅ | 4.10x |
| Kaby Lake (2017 MBP) | AVX2 | 39/39 ✅ | 59 | 54/54 ✅ | 3.49x |
| Mac Mini M1 | NEON | 39/39 ✅ | 59 | 54/54 ✅ | 2.31x |
| Asus TUF A16 (Windows) | AVX-512 | 39/39 ✅ | 59 | 54/54 ✅ | 1.64x |

All four machines validated. 16 distributions, 39/39 correctness tests, 54/54 SIMD tests.

**Total suite counts differ by machine:**
- Kaby Lake / M1 / Asus TUF A16 (59): full GTest + requires-expressions — all tests present.
- Ivy Bridge/Catalina (53): 6 timing-labelled tests excluded by
  `LIBSTATS_HAS_REQUIRES_EXPRESSIONS` gating (`test_exponential_enhanced`,
  `test_uniform_enhanced`, `test_poisson_enhanced`, `test_discrete_enhanced`,
  `test_gamma_enhanced`, and one pool variant). Correctness suite unaffected (39/39).
- Asus TUF A16 (Windows): GTest available via vcpkg `gtest:x64-windows 1.17.0`;
  all correctness-labelled GTest tests run correctly.

### SIMD Batch Operation Speedups (Ivy Bridge, AVX)
Vectorized batch kernels added to Exponential (PDF/LogPDF/CDF), Gamma (PDF/LogPDF), and Uniform (CDF).
All use the compute+fixup pattern documented in `src/gaussian.cpp` section 18.

| Distribution | Op | Speedup |
|---|---|---|
| Exponential | PDF | 10.5x |
| Exponential | LogPDF | 20.8x |
| Exponential | CDF | 10.1x |
| Gamma | PDF | 9.7x |
| Gamma | LogPDF | 7.1x |
| Uniform | CDF | 25.2x |

Overall `simd_verification` AVX speedup: 4.10x. 54/54 SIMD tests pass.

### Deferred Items
- AVX-512 transcendentals delegate to AVX (1.64x overall vs ~4x expected) — confirmed on AMD Ryzen 7 7445 (AVX-512, Windows);
  fix by ensuring simd_avx512.cpp routes exp/log through AVX-512 intrinsics rather than falling
  back to the AVX implementation; deferred post-v1.3.0
- Future: `vector_floor` + `vector_blend` primitives across all SIMD backends to enable
  branchless Discrete CDF and Uniform PDF/LogPDF; low priority given existing batch-path speedups
  (Discrete 8–15x, Uniform 39–54x) already achieved through amortization

### Distributions Added in v1.3.0
New distributions added in v1.3.0:
- **Binomial** — B(n, p); PMF via lgamma log-space; CDF via I_{1−p}(n−k, k+1);
  MLE closed-form p̂ = k̄/n̂. VECTORIZED = cached scalar loop.
- **Negative Binomial** — NB(r, p); real-valued r; PMF via lgamma; CDF via I_p(r, k+1);
  MLE: MoM seed + Newton–Raphson profile score using digamma/trigamma.
  Sampling: Gamma(r,(1−p)/p)-Poisson mixture.

Shared utility additions:
- `detail::trigamma(x)` added to `math_utils`: A&S §6.4.12, accuracy < 2×10⁻¹⁴.

Validation (v1.3.0, primary dev machine — Ivy Bridge AVX):
- correctness suite: 39/39 PASS
- `simd_verification`: 54/54 PASS, overall 4.10x (unchanged — discrete distributions use scalar loops)

### Distributions Added in v1.0.0
New distributions added in v1.0.0:
- **Student's t** — standalone implementation with SIMD log-space PDF/LogPDF and CDF via incomplete beta
- **Chi-squared** — delegation wrapper over Gamma(α=ν/2, β=1/2)
- **Beta** — standalone bounded-support distribution with two-log SIMD PDF/LogPDF and CDF via regularized incomplete beta

Shared utility additions:
- `detail::digamma(x)` promoted into `math_utils`
- `detail::inverse_beta_i(p, a, b)` added for Beta quantiles

Validation (v1.0.0):

Ivy Bridge AVX (primary dev machine):
- correctness suite: 34/34 PASS
- `simd_verification`: 54/54 PASS, overall 4.10x
- new-distribution speedups: Chi-squared PDF 9.5x/LogPDF 7.0x, Student's t PDF 7.3x/LogPDF 7.6x,
  Beta PDF 4.6x/LogPDF 4.4x

Asus TUF A16 (Windows, AVX-512 — first AVX-512 validation):
- correctness suite: 33/33 PASS (GTest available via vcpkg gtest:x64-windows 1.17.0)
- `simd_verification`: 54/54 PASS, overall 1.64x
- AVX-512 arithmetic/log-space paths: Gaussian LogPDF 21.9x, Exponential LogPDF 11.8x,
  Uniform LogPDF 7.5x — strong where transcendentals are not involved
- Overall speedup limited by transcendental delegation to AVX (see Deferred Items)

Kaby Lake AVX2 (2017 MBP):
- correctness suite: 33/33 PASS
- `simd_verification`: 54/54 PASS, overall 3.49x
- new-distribution speedups: Chi-squared PDF 13.8x/LogPDF 10.5x, Student's t PDF 6.3x/LogPDF 18.4x,
  Beta PDF 5.3x/LogPDF 4.1x

All four machines validated at v1.0.0.

### Development Ecosystem

| Machine | OS | CPU | SIMD | Notes |
|---|---|---|---|---|
| MacBook Pro 9,1 (2012) | macOS Catalina | Intel Ivy Bridge i7-3820QM | SSE2 + AVX | Primary dev machine |
| MacBook Pro 14,1 (2017) | macOS Ventura | Intel Kaby Lake | SSE2 + AVX + AVX2 + FMA | AVX2/FMA validation |
| Mac Mini M1 | macOS Tahoe | Apple Silicon M1 | NEON only | ARM/NEON path validation |
| Asus TUF A16 (2025) | Windows 11 Pro | AMD Ryzen 7 7445 (Zen 4) | SSE2 + AVX + AVX2 + **AVX-512** | Windows/MSVC + first AVX-512 machine |

**Note:** The Asus TUF A16 (Ryzen 7 7445, Zen 4) is the first machine in this ecosystem with AVX-512 support.
`simd_avx512.cpp` was first exercised there at v1.0.0; 54/54 SIMD tests pass.
The `test_simd_policy` AVX-512 string (`"AVX-512"`) was confirmed correct.

**Note:** If setting up a fresh Windows machine, the build environment must be configured from scratch; see the one-time setup notes below.

### Windows Session Setup

> **Windows tool paths vary** by installation method (direct installer, `winget`, `chocolatey`, Microsoft Store, etc.). The paths below are common defaults — adjust for your installation. VS Build Tools and full VS editions use different default directories; see the one-time setup notes below for alternatives and auto-detection.

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
- CMake ≥ 3.20 required. Install from https://cmake.org/download/, `winget install Kitware.CMake`, or `choco install cmake`.
- vcpkg for GTest: `git clone https://github.com/microsoft/vcpkg C:\vcpkg && C:\vcpkg\bootstrap-vcpkg.bat`. The path `C:\vcpkg` is a convention; if installed via `winget install Microsoft.vcpkg` or `choco install vcpkg` the location will differ — run `where vcpkg` to find it.
- Configure: `cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake`
  (adjust the toolchain path if vcpkg is not at `C:\vcpkg`)
- Build: `cmake --build . --config Release --parallel`
- GTest installed via vcpkg (`gtest:x64-windows 1.17.0`) — all 33 correctness tests pass

## Essential Build Commands

### Quick Build
```bash
# macOS/Linux — standard development build (default 'Dev' build type)
cmake -B build
cmake --build build --parallel   # equivalent to make -j$(nproc)
ctest --test-dir build --output-on-failure
```

Windows: use the commands in the `Windows Session Setup` section above.

### Common Build Configurations
```bash
# Development (default) - light optimization with debug info
cmake ..

# Production release - maximum optimization
cmake -DCMAKE_BUILD_TYPE=Release ..

# Full debugging support
cmake -DCMAKE_BUILD_TYPE=Debug ..

# Strict compiler warnings as errors (for compatibility testing)
cmake -DCMAKE_BUILD_TYPE=ClangStrict ..
cmake -DCMAKE_BUILD_TYPE=GCCStrict ..
cmake -DCMAKE_BUILD_TYPE=MSVCStrict ..
```

### Build System Features
- **Automatic parallel detection**: Detects CPU cores and configures optimal builds
- **Compiler detection**: Defaults to system AppleClang on macOS (ABI-safe); Homebrew LLVM available via `-DLIBSTATS_USE_HOMEBREW_LLVM=ON`
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

## Ad Hoc Compilation Outside CMake

For quick problem solving, diagnostics, and testing, you can compile directly without the CMake build system.

### macOS with Homebrew LLVM (Recommended)

#### Quick Template for Ad Hoc Tests
```bash
# Basic compilation template for libstats development
/opt/homebrew/opt/llvm/bin/clang++ -std=c++20 -stdlib=libc++ \
  -I/opt/homebrew/opt/llvm/include/c++/v1 \
  -I./include \
  -L/opt/homebrew/opt/llvm/lib/c++ \
  -L./build \
  -Wl,-rpath,/opt/homebrew/opt/llvm/lib/c++ \
  your_test.cpp -o test_output ./build/libstats.a
```

#### Path Detection for Different Mac Architectures
```bash
# For Apple Silicon Macs (ARM64)
LLVM_ROOT="/opt/homebrew/opt/llvm"

# For Intel Macs
LLVM_ROOT="/usr/local/opt/llvm"

# Auto-detect and compile
if [ -f "/opt/homebrew/opt/llvm/bin/clang++" ]; then
    LLVM_ROOT="/opt/homebrew/opt/llvm"
else
    LLVM_ROOT="/usr/local/opt/llvm"
fi

$LLVM_ROOT/bin/clang++ -std=c++20 -stdlib=libc++ \
  -I$LLVM_ROOT/include/c++/v1 \
  -I./include \
  -L$LLVM_ROOT/lib/c++ \
  -L./build \
  -Wl,-rpath,$LLVM_ROOT/lib/c++ \
  your_test.cpp -o test_output ./build/libstats.a
```

#### Essential Compiler Flags
- **`-std=c++20`**: C++20 standard required
- **`-stdlib=libc++`**: Use LLVM's modern C++ standard library
- **`-I$LLVM_ROOT/include/c++/v1`**: Modern C++ headers
- **`-L$LLVM_ROOT/lib/c++`**: Link against LLVM's libc++
- **`-Wl,-rpath,$LLVM_ROOT/lib/c++`**: Runtime library path
- **`-I./include`**: libstats header files

#### Static vs Dynamic Linking
```bash
# Static linking (recommended for ad hoc tests)
# Use the .a file directly - no runtime path issues
./build/libstats.a

# Dynamic linking (requires rpath setup)
-llibstats -Wl,-rpath,./build
```

#### Quick Test Template
Create `quick_test.cpp`:
```cpp
#include "libstats.h"
#include <iostream>

int main() {
    auto result = libstats::GaussianDistribution::create(0.0, 1.0);
    if (result.isOk()) {
        auto& gaussian = result.value;
        std::cout << "PDF at 0: " << gaussian.getProbability(0.0) << std::endl;
        std::cout << "CDF at 1: " << gaussian.getCumulativeProbability(1.0) << std::endl;
    }
    return 0;
}
```

Compile and run:
```bash
/opt/homebrew/opt/llvm/bin/clang++ -std=c++20 -stdlib=libc++ \
  -I/opt/homebrew/opt/llvm/include/c++/v1 \
  -I./include \
  -L/opt/homebrew/opt/llvm/lib/c++ \
  -Wl,-rpath,/opt/homebrew/opt/llvm/lib/c++ \
  quick_test.cpp -o quick_test ./build/libstats.a && ./quick_test
```

### System Compiler Fallback (Linux/Other Systems)
```bash
# Basic template for system compiler
g++ -std=c++20 -Wall -Wextra -O2 \
  -I./include \
  -L./build \
  your_test.cpp -o test_output -lstats

# Or with Clang
clang++ -std=c++20 -stdlib=libc++ \
  -I./include \
  -L./build \
  your_test.cpp -o test_output -lstats
```

### Troubleshooting Ad Hoc Compilation

#### Common Issues
- **`std::bad_function_call` undefined symbols**: Use `-stdlib=libc++` with Homebrew LLVM
- **Library not found**: Use static linking (`./build/libstats.a`) instead of `-llibstats`
- **C++20 features not available**: Ensure using Homebrew LLVM, not system compiler
- **Header not found**: Check `-I./include` path is correct

#### Verification Commands
```bash
# Check C++20 support
echo '__cplusplus' | /opt/homebrew/opt/llvm/bin/clang++ -std=c++20 -E -x c++ - | tail -n 1
# Should output: 202002 or higher

# Check which compiler you're using
which clang++
/opt/homebrew/opt/llvm/bin/clang++ --version
```

## Project Architecture

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

#### Statistical Distributions (9 implemented)
1. **Gaussian** (Normal) - N(μ, σ²)
2. **Exponential** - Exp(λ)
3. **Uniform** - U(a, b)
4. **Poisson** - P(λ)
5. **Discrete** - Custom discrete distributions
6. **Gamma** - Γ(α, β)
7. **Chi-squared** - χ²(ν) — delegation wrapper over Gamma(α=ν/2, β=1/2)
8. **Student's t** - t(ν) — SIMD log-space PDF/LogPDF and CDF via incomplete beta
9. **Beta** - Beta(α, β) — two-log SIMD PDF/LogPDF and CDF via regularized incomplete beta

Each provides: PDF/CDF/Quantiles, Statistical Moments, Parameter Estimation (MLE), Random Sampling, Statistical Validation, SIMD batch operations.

#### Platform Optimization
- **CPU Feature Detection**: Runtime SIMD capability detection
- **Threading Systems**: Comprehensive detection (TBB, OpenMP, pthreads, GCD, Windows Thread Pool)
- **Memory Management**: SIMD-aligned allocations and cache-aware algorithms

## Code Organization

### Header Architecture
```
include/
├── libstats.h              # Complete library (single include)
├── core/                   # Core mathematical and statistical components
│   ├── constants/          # Mathematical, precision, statistical constants
│   ├── distribution_*.h    # Distribution framework components
│   └── *_common.h         # Consolidated headers for faster compilation
├── distributions/          # Concrete distributions (gaussian.h, etc.)
└── platform/              # SIMD, threading, parallel execution
```

### Source Organization
```
src/
├── [Level 0-1] Foundation and utilities (cpu_detection.cpp, safety.cpp)
├── [Level 2] Platform capabilities (thread_pool.cpp, work_stealing_pool.cpp)
├── [Level 3] Infrastructure (benchmark.cpp, performance_dispatcher.cpp)
├── [Level 4] Framework (distribution_base.cpp)
└── [Level 5] Distributions (gaussian.cpp, exponential.cpp, etc.)
```

### Object Library Architecture
The CMake system uses dependency-aware object libraries for parallel compilation:
- `libstats_foundation_obj` → `libstats_core_utilities_obj` → `libstats_infrastructure_obj` → `libstats_framework_obj` → `libstats_distributions_obj`
- Enables optimal incremental builds and clear architectural boundaries

## Common Development Tasks

### Working with Distributions

#### Creating New Distributions
1. Use consolidated headers: `#include "../core/distribution_common.h"`
2. Follow the 24-section standardized template (see `exponential.h` as reference)
3. Implement the standardized test patterns (`*_basic.cpp` and `*_enhanced.cpp`)

#### Testing Strategy
- **All levels**: GTest-based tests registered with CTest
- Correctness tests: run `ctest -LE "timing|benchmark"` (parallel-safe)
- Timing tests: run `ctest -j1 -L timing` on a quiet machine
- **Coverage**: 23 correctness tests + timing/benchmark suite

### Performance Optimization

#### SIMD Development
- Use `libstats::simd::*` namespace for vectorized operations
- Runtime dispatch automatically selects best available instruction set
- Test with `./build/tools/simd_verification`

#### Parallel Processing
- Auto-dispatch API: `getProbability(std::span<const double>, std::span<double>, hint)`
- Explicit control: `getProbabilityWithStrategy(spans, Strategy::PARALLEL)`
- Dispatch thresholds are per-(architecture, distribution, operation) in `dispatch_thresholds.h`
- Thresholds derived from four-architecture profiling data in `data/profiles/dispatcher/`

### Build System Customization

#### CMake Options
```bash
# Enable verbose build messages for debugging
cmake -DLIBSTATS_VERBOSE_BUILD=ON ..

# Force TBB usage over platform-native threading
cmake -DLIBSTATS_FORCE_TBB=ON ..

# Conservative SIMD settings for compatibility
cmake -DLIBSTATS_CONSERVATIVE_SIMD=ON ..

# Disable tools or tests
cmake -DLIBSTATS_BUILD_TOOLS=OFF -DLIBSTATS_BUILD_TESTS=OFF ..
```

#### Compiler-Specific Builds
The build system supports cross-compiler compatibility testing with specialized build types that enable consistent warning levels across GCC, Clang, and MSVC.

## Important Guidelines

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

### Platform-Specific Notes
- **macOS**: System AppleClang is the default (ABI-safe for all consumers); use `-DLIBSTATS_USE_HOMEBREW_LLVM=ON` only when Homebrew libc++ is needed across the entire toolchain
- **Build artifacts**: Always in `build/tools/` and `build/tests/`, never `bin/`
- **Threading**: GCD preferred on macOS, TBB/OpenMP on Linux/Windows

## Testing and Validation

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

### Performance Validation
```bash
# Verify SIMD operations and performance
./build/tools/simd_verification

# Profile forced strategies for threshold tuning
./build/tools/strategy_profile

# System capability analysis
./build/tools/system_inspector --performance
```

The testing infrastructure ensures correctness across all optimization levels and provides regression detection for performance-critical paths.

## Warp Terminal Saved Workflows (warp.dev only)

> **Note for non-Warp users:** These workflows are available only in the Warp terminal. Users of other tools (Claude Code, Cursor, bare shells, etc.) should run the equivalent shell commands listed elsewhere in this file.

Saved workflows in `.warp/workflows/` are available directly in the Warp terminal for common tasks:

- **libstats: Clean Rebuild** — remove `build/` and rebuild from scratch; accepts `build_type` arg (default: `Dev`)
- **libstats: Validate Machine** — architecture detection, SIMD capabilities, correctness suite, and `simd_verification`; requires a current build
- **libstats: Switch Branch + Rebuild** — stash uncommitted changes, fetch, checkout target branch, pull, and clean rebuild in one step
- **libstats: Warning Audit** — build with a strict warning mode and display deduplicated warning counts; accepts `build_type` arg (default: `ClangWarn`)

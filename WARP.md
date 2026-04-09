# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

# libstats - Modern C++20 Statistical Distributions Library

## Project Overview

libstats is a high-performance C++20 statistical distributions library with enterprise-grade safety features, SIMD optimization, and zero external dependencies. The project is currently in Phase 5 (optimization and cross-platform tuning) after successfully delivering all core functionality.

**Current Status**: Core delivery complete ✅ - Focus on optimization, cross-platform compatibility, and performance tuning.

## Session Start: Architecture Detection

At the start of each libstats development session, verify the current machine architecture before making any SIMD-related decisions, reviewing test results, or adjusting thresholds.

```bash
# Identify the CPU architecture and OS
uname -m          # x86_64 = Intel/AMD | arm64 = Apple Silicon
uname -s          # Darwin = macOS | Linux | MINGW = Windows

# On macOS: identify the specific CPU
sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "(not macOS or not x86)"

# Check what SIMD the build detected (requires a current build)
./build/tools/system_inspector --quick 2>/dev/null || echo "Build not current — run cmake/make first"
```

### Why This Matters

The active SIMD level changes fundamentally between machines:

| Architecture | SIMD Available | Active simd_*.cpp files |
|---|---|---|
| Intel Mac (most) | SSE2, AVX (no AVX2) | `simd_sse2.cpp`, `simd_avx.cpp` |
| Intel Mac (Haswell+) | SSE2, AVX, AVX2 | + `simd_avx2.cpp` |
| Apple Silicon (ARM64) | NEON only | `simd_neon.cpp` |
| Linux x86 (typical) | SSE2, AVX, AVX2 | `simd_sse2.cpp`, `simd_avx.cpp`, `simd_avx2.cpp` |

SIMD code paths, performance thresholds, and test results are architecture-dependent. If the machine has changed since the last session:
- Note the change explicitly
- Verify the build directory is current for this architecture (`cmake ..` may be needed)
- Threshold values in `src/parallel_thresholds.cpp` may need review
- Benchmark results are not comparable across architectures

## Essential Build Commands

### Quick Build
```bash
# Standard development build (default 'Dev' build type)
mkdir build && cd build
cmake ..
make -j$(nproc)
ctest --output-on-failure
```

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
- **Compiler auto-detection**: Finds and configures Homebrew LLVM on macOS automatically
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
./build/tools/parallel_threshold_benchmark
./build/tools/simd_verification
./build/tools/performance_dispatcher_tool

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

#### Statistical Distributions (6 implemented)
1. **Gaussian** (Normal) - N(μ, σ²)
2. **Exponential** - Exp(λ)
3. **Uniform** - U(a, b)
4. **Poisson** - P(λ)
5. **Discrete** - Custom discrete distributions
6. **Gamma** - Γ(α, β)

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
├── [Level 2] Platform capabilities (thread_pool.cpp, parallel_thresholds.cpp)
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
- **Level 0-3**: Service-level tests with `main()` functions (fast, minimal dependencies)
- **Level 4-5**: GTest-based tests for complex statistical methods
- **Coverage**: 27 tests total (25 standalone + 2 GTest)

### Performance Optimization

#### SIMD Development
- Use `libstats::simd::*` namespace for vectorized operations
- Runtime dispatch automatically selects best available instruction set
- Test with `./build/tools/simd_verification`

#### Parallel Processing
- Auto-dispatch API: `getProbability(std::span<const double>, std::span<double>, hint)`
- Explicit control: `getProbabilityWithStrategy(spans, Strategy::PARALLEL)`
- Performance thresholds: <8 elements (scalar), 8-1000 (SIMD), >1000 (parallel)

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
- **macOS**: Homebrew LLVM preferred for C++20 execution policies
- **Build artifacts**: Always in `build/tools/` and `build/tests/`, never `bin/`
- **Threading**: GCD preferred on macOS, TBB/OpenMP on Linux/Windows

## Testing and Validation

### Running Tests
```bash
# Run all tests (timing assertions may be flaky under parallel load)
ctest --output-on-failure

# Correctness only — safe to run in parallel, excludes timing-sensitive assertions
ctest --output-on-failure -LE "timing|benchmark"

# Timing validation — run serially on a quiet machine for reliable results
ctest --output-on-failure -j1 -L timing

# Or via make targets
make run_tests          # Correctness suite (parallel-safe)
make run_tests_timing   # Timing suite (serial, quiet machine required)
make run_all_tests      # Everything

# Run a specific test
ctest -R test_gaussian_basic
ctest -R test_gaussian_enhanced  # Contains timing assertions

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

# Analyze parallel thresholds
./build/tools/parallel_threshold_benchmark

# System capability analysis
./build/tools/system_inspector --performance
```

The testing infrastructure ensures correctness across all optimization levels and provides regression detection for performance-critical paths.

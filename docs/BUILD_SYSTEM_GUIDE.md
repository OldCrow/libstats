# libstats Build System Guide

## Overview

This guide provides comprehensive information about the libstats build system, covering CMake configuration, build types, SIMD optimization, parallel builds, cross-platform considerations, and alternatives for building outside of CMake.

## Table of Contents

1. [Quick Start](#quick-start)
2. [CMake Build System](#cmake-build-system)
3. [Build Types Reference](#build-types-reference)
4. [SIMD Detection and Optimization](#simd-detection-and-optimization)
5. [Parallel Build Configuration](#parallel-build-configuration)
6. [Cross-Platform Support](#cross-platform-support)
7. [Advanced Configuration](#advanced-configuration)
8. [Building Outside CMake](#building-outside-cmake)
9. [Troubleshooting](#troubleshooting)

## Quick Start

### Basic Build
```bash
# Clone and build with default configuration
git clone https://github.com/YourOrg/libstats.git
cd libstats
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### Using Build Script
```bash
# Use the provided build script for automatic parallel detection
./scripts/build.sh Release
```

### Testing Build
```bash
# Run tests to verify everything works
ctest --output-on-failure
```

## CMake Build System

### Requirements

- **CMake**: 3.20 or later
- **C++20 Compiler**: Clang 14+, GCC 10+, MSVC 2019+
- **Platform**: Windows, macOS, Linux

### Key Features

- **Automatic Parallel Detection**: Detects CPU cores and configures optimal parallel builds
- **Compiler Detection**: Automatically detects and configures Homebrew LLVM on macOS
- **SIMD Optimization**: Robust runtime SIMD detection with fallbacks
- **Threading Systems**: Comprehensive detection of TBB, OpenMP, pthreads, GCD
- **Cross-Platform**: Native support for Windows, macOS, and Linux

### CMake Configuration Options

```bash
# Build type selection
cmake -DCMAKE_BUILD_TYPE=Release ..

# Verbose build messages for debugging
cmake -DLIBSTATS_VERBOSE_BUILD=ON ..

# Force TBB usage (override platform-native threading)
cmake -DLIBSTATS_FORCE_TBB=ON ..

# Enable runtime SIMD checks for cross-compilation
cmake -DLIBSTATS_ENABLE_RUNTIME_CHECKS=ON ..

# Conservative SIMD settings (disable newer instruction sets)
cmake -DLIBSTATS_CONSERVATIVE_SIMD=ON ..

# Disable tools build
cmake -DLIBSTATS_BUILD_TOOLS=OFF ..
```

## Build Types Reference

The build system provides multiple build types optimized for different use cases:

### Standard Build Types (Cross-Platform)

| Build Type | Purpose | Optimization | Debug Info | Warnings as Errors |
|------------|---------|--------------|------------|-------------------|
| **Dev** | Daily development (Default) | Light (-O1) | Yes | No |
| **Debug** | Full debugging | None (-O0) | Yes | No |
| **Release** | Production builds | Maximum (-O2/-O3) | No | No |

### Compiler-Specific Build Types

#### Clang Build Types
- **ClangStrict**: Strict Clang warnings as errors
- **ClangWarn**: Strict Clang warnings (not errors)

#### GCC Build Types
- **GCCStrict**: Strict GCC warnings as errors
- **GCCWarn**: Strict GCC warnings (not errors)

#### MSVC Build Types
- **MSVCStrict**: Strict MSVC warnings as errors
- **MSVCWarn**: Strict MSVC warnings (not errors)

### Usage Examples

```bash
# Default development build
cmake ..

# Production release build
cmake -DCMAKE_BUILD_TYPE=Release ..

# Strict compiler checking
cmake -DCMAKE_BUILD_TYPE=ClangStrict ..

# Performance testing
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

### Build Type Details

#### Dev (Default)
**Purpose**: Optimal for daily development work
- **MSVC**: `/W3 /O1 /Zi /MD`
- **Clang**: `-O1 -g -Wall -Wextra`
- **GCC**: `-O1 -g -Wall -Wextra`

#### Release
**Purpose**: Maximum performance for production
- **MSVC**: `/W3 /O2 /DNDEBUG`
- **Clang**: `-O3 -DNDEBUG -Wall`
- **GCC**: `-O3 -DNDEBUG -Wall`

#### ClangStrict
**Purpose**: Comprehensive Clang error checking
```cpp
-Wall -Wextra -Werror -pedantic
-Wconversion -Wfloat-conversion -Wimplicit-int-conversion
-Wdouble-promotion -Wsign-conversion -Wshadow
-Wcast-align -Wcast-qual -Wold-style-cast
-Woverloaded-virtual -Wmissing-declarations
```

## SIMD Detection and Optimization

### Robust SIMD System

The build system implements a dual-layer SIMD detection system:

1. **Compiler Support Check**: Verifies compiler can generate SIMD instructions
2. **Runtime CPU Check**: Actually executes SIMD instructions to ensure CPU support
3. **Architecture Awareness**: x86_64 vs ARM64 instruction set compatibility

### Supported SIMD Instruction Sets

#### x86_64 Platforms
- **SSE2**: Baseline (always available on 64-bit)
- **AVX**: 256-bit vector operations
- **AVX2**: Enhanced 256-bit integer operations
- **AVX-512**: 512-bit vector operations (server CPUs)

#### ARM64 Platforms
- **NEON**: ARM's SIMD instruction set

### SIMD Detection Examples

#### Modern CPU (Full SIMD Support)
```
-- Runtime sse2 test: PASSED
-- SIMD: SSE2 enabled (compiler + runtime)
-- Runtime avx test: PASSED
-- SIMD: AVX enabled (compiler + runtime)
-- Runtime avx2 test: PASSED
-- SIMD: AVX2 enabled (compiler + runtime)
-- SIMD detection complete:
--   SSE2: TRUE, AVX: TRUE, AVX2: TRUE, AVX-512: FALSE
```

#### Apple Silicon (ARM64)
```
-- SIMD: SSE2 disabled (compiler not supported)
-- SIMD: AVX disabled (compiler not supported)
-- SIMD: NEON available on AArch64
-- Runtime neon test: PASSED
-- SIMD: NEON enabled (compiler + runtime)
-- SIMD detection complete:
--   NEON: TRUE, SSE2/AVX: FALSE (ARM64 architecture)
```

### Cross-Compilation SIMD Override

For cross-compilation environments:

```bash
# Force enable/disable specific SIMD features
export LIBSTATS_FORCE_SSE2=1
export LIBSTATS_FORCE_AVX=0
export LIBSTATS_FORCE_AVX2=1
export LIBSTATS_FORCE_AVX512=0
export LIBSTATS_FORCE_NEON=1

cmake ..
make -j$(nproc)
```

### SIMD Performance Impact

- **Function Pointer Lookup**: ~1-2 CPU cycles
- **Runtime Dispatch**: One-time initialization cost
- **Memory Overhead**: Minimal (few function pointers)
- **Performance Gains**: 2-70x speedup depending on operation complexity

## Parallel Build Configuration

### Automatic Parallel Detection

The build system automatically detects CPU cores and configures optimal parallel builds:

- **Cross-Platform Detection**: Uses `ProcessorCount()` CMake module
- **Environment Variables**: Sets `MAKEFLAGS=-j{CPU_COUNT}` and `CMAKE_BUILD_PARALLEL_LEVEL`
- **Fallback**: Defaults to 4 parallel jobs if detection fails
- **Zero Configuration**: Optimal performance out of the box

### Build Script Usage

The comprehensive build script provides automatic parallel detection:

```bash
# Basic usage with auto-detected parallel jobs
./scripts/build.sh

# Specific build type with parallelism
./scripts/build.sh Release

# Override parallel job count
./scripts/build.sh -j 16 Release

# Clean build with tests
./scripts/build.sh -c -t Release

# Configure only (no build)
./scripts/build.sh --configure-only Dev

# Verbose output with parallel builds
./scripts/build.sh -v Release
```

### Manual Parallel Build Commands

```bash
# CMake with automatic parallel detection
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --parallel  # Uses CMAKE_BUILD_PARALLEL_LEVEL

# Make with detected parallel jobs
make  # Automatically uses MAKEFLAGS=-j{CPU_COUNT}

# Override manually
cmake --build . --parallel 8
make -j8
```

### Performance Benefits

- **Utilizes All CPU Cores**: Automatic detection and usage
- **4-8x Faster Builds**: On multi-core systems
- **Cross-Platform**: Works on Windows, macOS, Linux
- **Zero Configuration**: Optimal performance by default

## Cross-Platform Support

### Platform-Specific Configurations

#### macOS
- **Compiler Detection**: Automatic Homebrew LLVM detection with fallback to system Clang
- **SIMD Support**: NEON (Apple Silicon) or x86_64 SIMD (Intel)
- **Threading**: Grand Central Dispatch (GCD) with TBB/OpenMP fallbacks
- **C++ Standard Library**: Automatic LLVM libc++ configuration for C++20 features

#### Linux
- **Compilers**: GCC, Clang support with automatic detection
- **SIMD Support**: Full x86_64 SIMD detection (SSE2, AVX, AVX2, AVX-512)
- **Threading**: POSIX threads, TBB, OpenMP detection
- **Distribution**: Supports major Linux distributions

#### Windows
- **Compilers**: MSVC, ClangCL support
- **SIMD Support**: x86_64 SIMD instruction sets
- **Threading**: Windows Thread Pool API detection
- **Visual Studio Integration**: Full integration with VS build system

### Cross-Compilation Support

```bash
# Enable runtime checks for cross-compilation
cmake -DLIBSTATS_ENABLE_RUNTIME_CHECKS=ON ..

# Override SIMD detection for target CPU
export LIBSTATS_FORCE_AVX2=1
export LIBSTATS_FORCE_AVX512=0

# Conservative build for older hardware
cmake -DLIBSTATS_CONSERVATIVE_SIMD=ON ..
```

## Advanced Configuration

### CMake Generator Expressions

The build system uses modern CMake generator expressions for better cross-platform control:

```cmake
# Platform-specific compiler flags
target_compile_options(libstats_static PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
    $<$<CXX_COMPILER_ID:Clang>:-Wall -Wextra>
    $<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra>
)

# Configuration-specific options
target_compile_definitions(libstats_static PRIVATE
    $<$<CONFIG:Debug>:LIBSTATS_DEBUG_MODE>
    $<$<CONFIG:Release>:NDEBUG>
)
```

### Threading System Detection

The build system performs comprehensive threading capability detection:

- **Intel TBB**: Optimal for C++20 parallel execution policies
- **OpenMP**: Cross-platform parallel programming
- **POSIX Threads**: Unix/Linux standard threading
- **Grand Central Dispatch**: macOS-native high-performance threading
- **Windows Thread Pool**: Windows-native threading API

### Homebrew LLVM Integration (macOS)

Automatic detection and configuration of Homebrew LLVM:

```bash
# ARM64 Macs (Apple Silicon)
/opt/homebrew/opt/llvm

# Intel Macs
/usr/local/opt/llvm
```

Features enabled with Homebrew LLVM:
- **C++20 Execution Policies**: Enhanced PSTL support
- **Modern libc++**: Latest C++ standard library features
- **Better Diagnostics**: Superior error messages and warnings
- **Performance**: Optimized code generation

### Cache System

The build system implements intelligent caching to avoid repeated detection:

- **Threading Detection**: Cached across reconfiguration
- **SIMD Capabilities**: Cached runtime test results
- **Compiler Detection**: Cached Homebrew LLVM detection
- **Performance**: Faster subsequent CMake runs

## Building Outside CMake

For developers who prefer alternative build systems or need custom integration, here are the key considerations:

### Compiler Requirements

#### C++20 Support Required
```bash
# Minimum compiler versions:
# - Clang 14+ with libc++14+ or libstdc++9+
# - GCC 10+ with libstdc++9+
# - MSVC 2019+ (19.20+)

# Verify C++20 support:
echo '__cplusplus' | clang++ -std=c++20 -E -x c++ - | tail -n 1
# Should output: 202002 or higher
```

### Essential Compiler Flags

#### Universal Flags (All Platforms)
```bash
-std=c++20
-Wall -Wextra
-I./include
-DLIBSTATS_VERSION_MAJOR=0
-DLIBSTATS_VERSION_MINOR=8
-DLIBSTATS_VERSION_PATCH=3
```

#### Platform-Specific Flags

**macOS with Homebrew LLVM:**
```bash
/opt/homebrew/opt/llvm/bin/clang++ -std=c++20 -stdlib=libc++ \
  -I/opt/homebrew/opt/llvm/include/c++/v1 \
  -L/opt/homebrew/opt/llvm/lib/c++ \
  -Wl,-rpath,/opt/homebrew/opt/llvm/lib/c++ \
  -D_LIBCPP_HAS_EXPERIMENTAL_PSTL=1
```

**Linux with GCC:**
```bash
g++ -std=c++20 -pthread -fPIC \
  -march=native -O3 -DNDEBUG
```

**Windows with MSVC:**
```bash
cl.exe /std:c++20 /EHsc /W3 /O2 \
  /DNOMINMAX /D_USE_MATH_DEFINES
```

### SIMD Compilation

#### Manual SIMD Flags
```bash
# SSE2 (baseline)
-msse2

# AVX support
-mavx

# AVX2 support
-mavx2

# AVX-512 support (server CPUs)
-mavx512f

# ARM NEON (Apple Silicon/ARM64)
-mfpu=neon  # Only on ARM targets
```

#### Architecture-Specific Considerations

**x86_64 Build:**
```bash
# Enable all available SIMD for native CPU
-march=native

# Or specific instruction sets
-msse2 -mavx -mavx2
```

**ARM64 Build (Apple Silicon):**
```bash
# NEON is available by default on ARM64, no special flags needed
# DO NOT use x86 SIMD flags (-mavx, -msse2) on ARM64
```

### Threading Configuration

#### Required Threading Libraries

**POSIX Systems (Linux/macOS):**
```bash
-pthread
```

**TBB Integration (Optional, Recommended):**
```bash
# Link TBB if available
-ltbb
```

**OpenMP (Alternative):**
```bash
# GCC/Clang
-fopenmp

# Link OpenMP runtime
-lgomp    # GCC
-lomp     # Clang
```

### Source File Organization

#### Core Library Files
```bash
# Always required
src/libstats_core.cpp
src/distributions/*.cpp

# SIMD implementations (conditional)
src/simd_fallback.cpp     # Always
src/simd_sse2.cpp        # If SSE2 available
src/simd_avx.cpp         # If AVX available
src/simd_avx2.cpp        # If AVX2 available
src/simd_neon.cpp        # If NEON available (ARM64)
```

### Library Linking

#### Static Library Creation
```bash
# Compile object files
clang++ -std=c++20 -c -I./include src/*.cpp

# Create static library
ar rcs libstats.a *.o
ranlib libstats.a
```

#### Dynamic Library Creation

**Linux:**
```bash
g++ -shared -fPIC -o libstats.so.0.8.3 *.o
ln -sf libstats.so.0.8.3 libstats.so
```

**macOS:**
```bash
clang++ -dynamiclib -o libstats.0.8.3.dylib *.o
ln -sf libstats.0.8.3.dylib libstats.dylib
```

**Windows:**
```bash
link.exe /DLL /OUT:libstats.dll *.obj
```

### Feature Detection Without CMake

#### Compile-Time Feature Detection
```cpp
// Check for C++20 execution policy support
#ifdef __has_include
  #if __has_include(<execution>) && __cplusplus >= 202002L
    #define LIBSTATS_HAS_EXECUTION_POLICY 1
  #endif
#endif

// Check for TBB
#ifdef __has_include
  #if __has_include(<tbb/tbb.h>)
    #define LIBSTATS_HAS_TBB 1
  #endif
#endif
```

#### Runtime SIMD Detection
```cpp
// Basic CPUID-based detection (x86_64)
#ifdef __x86_64__
bool has_avx2() {
    unsigned int eax, ebx, ecx, edx;
    __get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx);
    return (ebx & bit_AVX2) != 0;
}
#endif
```

### Build System Examples

#### Makefile Example
```makefile
CXX = clang++
CXXFLAGS = -std=c++20 -O3 -Wall -Wextra -march=native
INCLUDES = -I./include
LDFLAGS = -pthread

SOURCES = $(wildcard src/*.cpp src/*/*.cpp)
OBJECTS = $(SOURCES:.cpp=.o)
TARGET = libstats.a

all: $(TARGET)

$(TARGET): $(OBJECTS)
	ar rcs $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET)
```

#### Ninja Example
```ninja
cxx = clang++
cxxflags = -std=c++20 -O3 -Wall -Wextra -march=native
includes = -I./include

rule cxx
  command = $cxx $cxxflags $includes -c $in -o $out

rule ar
  command = ar rcs $out $in

build libstats.a: ar src/core.o src/distributions/gaussian.o
build src/core.o: cxx src/core.cpp
build src/distributions/gaussian.o: cxx src/distributions/gaussian.cpp
```

### Integration Considerations

- **Header-Only Alternative**: Consider header-only mode for simpler integration
- **Package Managers**: Consider using Conan, vcpkg, or similar for dependency management
- **Testing**: Ensure to test SIMD dispatch and threading on target platforms
- **Performance**: Profile your specific build configuration vs CMake build
- **Maintenance**: CMake build system receives updates; manual builds require maintenance

## Troubleshooting

### Common Build Issues

#### 1. CMake Version Too Old
```bash
# Error: CMake 3.20 or higher is required
# Solution: Update CMake
sudo apt-get update && sudo apt-get install cmake  # Linux
brew install cmake  # macOS
```

#### 2. C++20 Not Supported
```bash
# Error: The compiler does not support C++20
# Solution: Update compiler or use Homebrew LLVM on macOS
brew install llvm  # macOS
```

#### 3. SIMD Detection Issues
```bash
# Debug SIMD detection
cmake -DLIBSTATS_VERBOSE_BUILD=ON ..

# Force specific SIMD settings
export LIBSTATS_FORCE_AVX2=0
export LIBSTATS_FORCE_AVX512=0
cmake ..
```

#### 4. Parallel Build Memory Issues
```bash
# Reduce parallel jobs if running out of memory
./scripts/build.sh -j 4 Release

# Or build sequentially
./scripts/build.sh -j 1 Release
```

#### 5. Threading System Issues
```bash
# Check detected threading systems
cmake -DLIBSTATS_VERBOSE_BUILD=ON ..

# Force TBB usage
cmake -DLIBSTATS_FORCE_TBB=ON ..
```

### Platform-Specific Issues

#### macOS Issues

**Homebrew LLVM Not Found:**
```bash
# Verify Homebrew LLVM installation
ls -la /opt/homebrew/opt/llvm/bin/clang++  # Apple Silicon
ls -la /usr/local/opt/llvm/bin/clang++     # Intel

# Reinstall if needed
brew uninstall llvm && brew install llvm
```

**Execution Policy Issues:**
```bash
# Enable experimental PSTL support
cmake -DLIBSTATS_VERBOSE_BUILD=ON ..
# Look for: "Enhanced with LLVM experimental PSTL"
```

#### Linux Issues

**Missing Development Packages:**
```bash
# Install required development packages
sudo apt-get install build-essential cmake libtbb-dev  # Ubuntu/Debian
sudo yum install gcc-c++ cmake tbb-devel              # RHEL/CentOS
```

**GLIBC Version Issues:**
```bash
# Check GLIBC version
ldd --version

# Use older build type if needed
cmake -DCMAKE_BUILD_TYPE=Debug ..
```

#### Windows Issues

**MSVC Not Found:**
```bash
# Use Developer Command Prompt for VS
# Or specify compiler explicitly
cmake -G "Visual Studio 16 2019" -A x64 ..
```

**Missing Windows SDK:**
```bash
# Install Windows SDK via Visual Studio Installer
# Or use Windows 10 SDK standalone installer
```

### Debug Information

#### Verbose Build Output
```bash
# Enable verbose CMake output
cmake -DCMAKE_MESSAGE_LOG_LEVEL=STATUS ..
cmake -DLIBSTATS_VERBOSE_BUILD=ON ..

# Verbose make output
make VERBOSE=1

# Or with CMake
cmake --build . --verbose
```

#### CPU and SIMD Testing
```bash
# Build and run CPU test utility
cmake ..
make -j$(nproc)
./tools/cpu_test

# Test specific SIMD features
./tools/cpu_test sse2 avx avx2
```

#### Build Script Debugging
```bash
# Run build script with shell debugging
bash -x scripts/build.sh Release

# Check build script permissions
chmod +x scripts/build.sh
```

### Performance Validation

#### Verify Optimization Flags
```bash
# Check that optimization flags are applied
cmake -DCMAKE_BUILD_TYPE=Release ..
make VERBOSE=1 | grep -E "O2|O3|march"
```

#### Benchmark Different Configurations
```bash
# Compare build types
./scripts/build.sh Release
./tests/performance_test > release_results.txt

./scripts/build.sh Debug
./tests/performance_test > debug_results.txt
```

## Conclusion

The libstats build system provides a robust, cross-platform solution that automatically detects and configures:

- **Optimal Parallel Builds**: Automatic CPU detection and parallel job configuration
- **SIMD Optimization**: Runtime-safe SIMD instruction set detection and dispatch
- **Threading Systems**: Comprehensive detection of TBB, OpenMP, pthreads, and native APIs
- **Compiler Support**: Automatic Homebrew LLVM detection with system compiler fallbacks
- **Cross-Platform**: Native support for Windows, macOS, and Linux

The CMake system is the recommended approach for most users, providing zero-configuration optimal builds. For specialized requirements, the manual build information enables custom integration while maintaining the performance and safety features of the library.

---

**Document Version**: 1.0
**Last Updated**: 2025-08-13
**Covers**: Complete build system, CMake configuration, SIMD detection, parallel builds, cross-platform support
**Replaces**: `build_types.md`, `parallel_builds.md`, `simd_build_system.md`

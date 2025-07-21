# Robust SIMD Build System

This document describes libstats' robust SIMD detection and build system that ensures compatibility across different CPUs and build environments.

## Problem Statement

Traditional SIMD detection in CMake has several issues:
- **Compiler capability ≠ CPU capability**: A compiler can generate AVX-512 code, but the target CPU might not support it
- **Build machine ≠ Runtime machine**: Code built on a modern machine might run on older hardware
- **Cross-compilation challenges**: Runtime checks can't be performed when cross-compiling
- **Illegal instruction errors**: Using unsupported instructions crashes the program

## Solution Overview

Our SIMD detection system performs both **compiler checks** and **runtime CPU checks**:

1. **Compiler Support**: Verifies the compiler can generate specific SIMD instructions
2. **Runtime Testing**: Actually executes SIMD instructions to ensure CPU support
3. **Cross-compilation Support**: Provides environment variable overrides for cross-compilation
4. **Runtime Dispatch**: Uses the detected capabilities to dispatch to appropriate implementations

## Build Configuration

### Basic Usage

```bash
# Configure and build with automatic SIMD detection
cmake ..
make -j$(nproc)

# Test CPU capabilities
./tools/cpu_test
```

### Advanced Options

```bash
# Enable runtime checks even when cross-compiling
cmake -DLIBSTATS_ENABLE_RUNTIME_CHECKS=ON ..

# Use conservative SIMD settings (disable newer instruction sets)
cmake -DLIBSTATS_CONSERVATIVE_SIMD=ON ..

# Disable tools build
cmake -DLIBSTATS_BUILD_TOOLS=OFF ..
```

## Cross-Compilation Support

When cross-compiling, you can override SIMD detection using environment variables:

```bash
# Force enable specific SIMD features
export LIBSTATS_FORCE_SSE2=1
export LIBSTATS_FORCE_AVX=0      # Disable AVX
export LIBSTATS_FORCE_AVX2=1     # Enable AVX2
export LIBSTATS_FORCE_AVX512=0   # Disable AVX-512
export LIBSTATS_FORCE_NEON=1     # Enable NEON (ARM)

# Then configure and build
cmake ..
make -j$(nproc)
```

## SIMD Detection Process

### Architecture-Specific Behavior

**Important**: The SIMD detection system is architecture-aware and optimized for native builds:

- **On x86_64**: Tests SSE2, AVX, AVX2, AVX-512 support; NEON is disabled
- **On ARM64/AArch64**: Tests NEON support; x86 SIMD is disabled (compiler rejects `-mavx2` etc. for ARM64 target)
- **Cross-compilation**: Use `LIBSTATS_FORCE_*` environment variables to override detection

This means:
- Building on Apple Silicon (ARM64) will show "SIMD: SSE2/AVX disabled (compiler not supported)" - this is correct behavior
- Building on Intel/AMD (x86_64) will show "SIMD: NEON disabled (not ARM architecture)" - this is also correct
- The compiler correctly rejects incompatible SIMD flags for the target architecture

### 1. Compiler Support Check

For each SIMD instruction set compatible with the target architecture:
1. Test if compiler accepts the corresponding flag (`-mavx2`, `-mavx512f`, `-mfpu=neon`, etc.)
2. Try to compile a simple test program using SIMD intrinsics
3. If both succeed, mark as "compiler supported"

### 2. Runtime CPU Check

For compiler-supported instruction sets:
1. Create and compile a runtime test program
2. Execute the test program with SIMD instructions
3. If execution succeeds without crashing, mark as "runtime supported"
4. Only runtime-supported instruction sets are enabled

### 3. Source File Selection

Based on detection results:
- Always include: `simd_fallback.cpp`, `simd_dispatch.cpp`
- Conditionally include: `simd_sse2.cpp`, `simd_avx.cpp`, `simd_avx2.cpp`, `simd_avx512.cpp`, `simd_neon.cpp`
- Each SIMD source file is compiled with appropriate compiler flags

## Build Examples

### Example 1: Modern CPU (AVX-512 Support)

```
-- Runtime sse2 test: PASSED
-- SIMD: SSE2 enabled (compiler + runtime)
-- Runtime avx test: PASSED  
-- SIMD: AVX enabled (compiler + runtime)
-- Runtime avx2 test: PASSED
-- SIMD: AVX2 enabled (compiler + runtime)
-- Runtime avx512 test: PASSED
-- SIMD: AVX-512 enabled (compiler + runtime)
-- SIMD detection complete:
--   SSE2: TRUE
--   AVX:  TRUE
--   AVX2: TRUE
--   AVX-512: TRUE
--   NEON: FALSE
```

### Example 2: Older CPU (SSE2 Only)

```
-- Runtime sse2 test: PASSED
-- SIMD: SSE2 enabled (compiler + runtime)
-- Runtime avx test: CRASHED (illegal instruction?)
-- SIMD: AVX disabled (runtime check failed)
-- Runtime avx2 test: CRASHED (illegal instruction?)
-- SIMD: AVX2 disabled (runtime check failed)
-- SIMD detection complete:
--   SSE2: TRUE
--   AVX:  FALSE
--   AVX2: FALSE
--   AVX-512: FALSE
--   NEON: FALSE
```

### Example 3: Apple Silicon (ARM64)

```
-- SIMD: SSE2 disabled (compiler not supported)
-- SIMD: AVX disabled (compiler not supported)
-- SIMD: AVX2 disabled (compiler not supported)
-- SIMD: AVX-512 disabled (compiler not supported)
-- SIMD: NEON available on AArch64 (no special flags needed)
-- Runtime neon test: PASSED
-- SIMD: NEON enabled (compiler + runtime)
-- SIMD detection complete:
--   SSE2: FALSE
--   AVX:  FALSE
--   AVX2: FALSE
--   AVX-512: FALSE
--   NEON: TRUE
```

### Example 4: Cross-Compilation

```
-- Cross-compiling detected - skipping runtime SIMD checks
-- Use LIBSTATS_ENABLE_RUNTIME_CHECKS=ON to force runtime checks
-- Use environment variables LIBSTATS_FORCE_* to override SIMD settings
-- SIMD: SSE2 enabled (compiler only - cross-compiling)
```

## CPU Test Utility

The `cpu_test` utility provides detailed information about detected CPU capabilities:

```bash
./tools/cpu_test
```

Example output:
```
LibStats CPU Capability Test
============================

CPU Information:
  Vendor: GenuineIntel
  Brand: Intel(R) Core(TM) i7-3820QM CPU @ 2.70GHz
  Family: 6
  Model: 58
  Stepping: 9

SIMD Capabilities:
  SSE2:     YES
  SSE3:     YES
  SSSE3:    YES
  SSE4.1:   YES
  SSE4.2:   YES
  AVX:      YES
  FMA:      NO
  AVX2:     NO
  AVX-512F: NO
  NEON:     NO

Optimal SIMD Configuration:
  Best Level: AVX
  Double Width: 4 elements
  Float Width: 8 elements
  Alignment: 32 bytes

Cache Information:
  L1 Data: 32 KB
  L1 Instruction: 32 KB
  L2: 256 KB
  L3: 8192 KB

CPU Topology:
  Logical Cores: 8
  Physical Cores: 4
  Hyperthreading: YES

Feature Consistency: VALID

CPU capability test completed successfully.
```

You can also test specific features:
```bash
./tools/cpu_test sse2 avx avx2
```

## Integration with Build Systems

### CMake Integration

When using libstats as a dependency:

```cmake
find_package(libstats REQUIRED)
target_link_libraries(my_target PRIVATE libstats::static)

# Access SIMD capabilities detected at build time
if(libstats_HAS_AVX2)
    message(STATUS "libstats was built with AVX2 support")
endif()
```

### Environment Variables

For build automation:

```bash
# Conservative build (older hardware)
export LIBSTATS_FORCE_AVX2=0
export LIBSTATS_FORCE_AVX512=0

# Optimized build (modern hardware)
export LIBSTATS_FORCE_AVX2=1
export LIBSTATS_FORCE_AVX512=1

cmake .. && make -j$(nproc)
```

## Troubleshooting

### Common Issues

1. **"illegal instruction" errors**
   - The old system was detecting compiler support but not runtime support
   - New system prevents this by testing actual execution

2. **Cross-compilation failures**
   - Use `LIBSTATS_ENABLE_RUNTIME_CHECKS=OFF` (default when cross-compiling)
   - Set `LIBSTATS_FORCE_*` environment variables for target CPU

3. **Conservative builds needed**
   - Use `LIBSTATS_CONSERVATIVE_SIMD=ON`
   - Manually disable newer instruction sets with environment variables

### Debug Information

Enable verbose CMake output to see detailed SIMD detection:
```bash
cmake -DCMAKE_MESSAGE_LOG_LEVEL=STATUS ..
```

## Performance Impact

The runtime dispatch system adds minimal overhead:
- Function pointer lookup: ~1-2 CPU cycles
- Initialization cost: One-time during library load
- Memory overhead: Negligible (few function pointers)

Benefits far outweigh costs:
- **Safety**: No illegal instruction crashes
- **Portability**: Same binary runs on different CPUs
- **Performance**: Optimal code path selected automatically

## Future Enhancements

Planned improvements:
- ARM SVE detection and support
- Intel AMX (Advanced Matrix Extensions) support
- GPU/OpenCL fallback detection
- Profile-guided optimization integration
- Benchmark-driven SIMD selection

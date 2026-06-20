# AVX-512 Testing Guide

This guide describes how libstats validates AVX-512 compile-time and runtime behaviour.

## Scope

AVX-512 support is validated in two layers:

1. **CI compile validation** — verifies AVX-512 sources compile with supported toolchains.
2. **Real hardware validation** — runs correctness and SIMD verification on an AVX-512-capable machine.

GitHub-hosted runners do not guarantee AVX-512 runtime hardware.

## Build validation

Configure and build in Release mode:

```bash
cmake -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release --parallel
```

Check that `simd_avx512.cpp` is included in the SIMD detection summary when the compiler and configuration enable it.

## Runtime validation

On an AVX-512-capable system:

```bash
./build-release/tools/system_inspector --quick
ctest --test-dir build-release --output-on-failure -LE "timing|benchmark"
./build-release/tools/simd_verification
```

Record:

- CPU model
- OS version
- compiler version
- active SIMD capabilities
- correctness test count
- `simd_verification` summary

## Expected v2.x validation target

The current project AVX-512 validation target is an AMD Zen 4 system. Other AVX-512 systems are valid but may show different speedups due to width, frequency, and instruction implementation differences.

## Common issues

### Compile error: AVX-512 intrinsic not available

Check compiler flags and SIMD detection output. AVX-512 sources require compiler support for AVX-512F and related extensions used by the backend.

### Runtime illegal instruction

This indicates a runtime dispatch bug or an executable built with CPU-specific flags that were not guarded correctly. Run `system_inspector --quick` and verify dispatch chooses AVX-512 only when supported by the CPU.

### Performance lower than AVX2

AVX-512 may downclock or have different throughput characteristics on some CPUs. Correctness matters first; speedups should be interpreted per machine.

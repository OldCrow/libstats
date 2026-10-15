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

On Windows, both cl.exe and clang-cl get `/arch:AVX512` globally on an
AVX-512 build host. cl.exe is the correctness build. clang-cl is the
performance build and is required for timing and threshold work: every
timing and threshold capture on Zen 4 uses it. Build it in its own
directory with Ninja:

```powershell
cmake -B build-clangcl -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=clang-cl
cmake --build build-clangcl --parallel
```

`/arch:AVX512` enables AVX-512 F, CD, BW, DQ and VL; clang-cl's `-mavx512f`
would enable F alone.

A timing run on Zen 4 must opt every measured process out of Windows power
throttling and launch it directly, not through ctest (a child does not
inherit the opt-out); otherwise a locked screen drops the clock and the
result reads like a dispatch-threshold anomaly: `docs/VALIDATION_HISTORY.md`, "Zen 4 timing:
power throttling"; helper
`docs/bench-evidence/2026-10-05-zen4-msvc-vs-clangcl/nothrottle.ps1`.

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

Runtime dispatch selects the AVX-512 tier only when the CPU reports both
AVX-512F and AVX-512DQ (`src/simd_policy.cpp`): `simd_avx512.cpp` uses DQ
instructions. A CPU with F but not DQ falls back to the next tier down.

## Expected v2.x validation target

The current project AVX-512 validation target is an AMD Zen 4 system. Other AVX-512 systems are valid but may show different speedups due to width, frequency, and instruction implementation differences.

## Common issues

### Compile error: AVX-512 intrinsic not available

Check compiler flags and SIMD detection output. AVX-512 sources require compiler support for AVX-512F and AVX-512DQ.

### Runtime illegal instruction

This indicates a runtime dispatch bug or an executable built with CPU-specific flags that were not guarded correctly. Run `system_inspector --quick` and verify dispatch chooses AVX-512 only when supported by the CPU.

### Performance lower than AVX2

AVX-512 may downclock or have different throughput characteristics on some CPUs. Correctness matters first; speedups should be interpreted per machine.

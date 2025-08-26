# AVX-512 Testing Guide for libstats

This guide explains how to set up comprehensive AVX-512 testing for the libstats project, including both compilation verification and runtime testing.

## Current AVX-512 Testing Status

### ✅ What We Test Now
- **NEON**: macOS (Apple Silicon) - both debug and release
- **SSE2**: Linux GCC, Linux Clang, Windows MSVC
- **AVX**: Linux GCC, Linux Clang, Windows MSVC
- **AVX2**: Linux GCC, Linux Clang, Windows MSVC

### ❌ What We're Missing
- **AVX-512**: No current testing in GitHub Actions workflows

## AVX-512 Testing Solutions

### Option 1: Self-Hosted Runner (Recommended for Production)

This provides the most comprehensive testing but requires access to AVX-512 hardware.

#### Hardware Requirements
- Intel CPU with AVX-512 support:
  - Skylake-X/SP (2017+)
  - Cascade Lake (2019+)
  - Ice Lake (2019+)
  - Tiger Lake (2020+)
  - Rocket Lake (2021+)
  - Alder Lake (2021+)
  - Sapphire Rapids (2023+)

#### Setup Steps

1. **Set up self-hosted runner**:
   ```bash
   # On your AVX-512 capable machine
   mkdir actions-runner && cd actions-runner
   curl -o actions-runner-linux-x64-2.311.0.tar.gz -L https://github.com/actions/runner/releases/download/v2.311.0/actions-runner-linux-x64-2.311.0.tar.gz
   tar xzf ./actions-runner-linux-x64-2.311.0.tar.gz
   ./config.sh --url https://github.com/OldCrow/libstats --token YOUR_TOKEN
   ./run.sh
   ```

2. **Update workflow** in `.github/workflows/avx512-testing.yml`:
   ```yaml
   runs-on: self-hosted-avx512  # Your runner label
   ```

3. **Benefits**:
   - ✅ Full runtime AVX-512 testing
   - ✅ Performance benchmarking with real AVX-512 workloads
   - ✅ Complete validation of AVX-512 code paths

### Option 2: Compilation Verification (Currently Implemented)

Tests that AVX-512 code compiles correctly without runtime execution.

#### What It Tests
- ✅ AVX-512 headers parse correctly
- ✅ AVX-512 intrinsics compile with GCC and Clang
- ✅ No compilation errors in AVX-512 code paths
- ✅ Proper linking of AVX-512 object files

#### Limitations
- ❌ Cannot test runtime behavior
- ❌ Cannot verify performance characteristics
- ❌ Cannot test CPU feature detection accuracy

#### Usage
The `avx512-compilation.yml` workflow runs automatically and will:
```bash
# Test compilation with various flag combinations
cmake -DCMAKE_CXX_FLAGS="-mavx512f -mavx512dq -mavx512bw -mavx512vl"
cmake --build build --parallel

# Verify AVX-512 symbols in object files
objdump -d *.o | grep -i "avx512\\|zmm"
```

### Option 3: Enhanced CMake Detection

#### Features
- Automatic AVX-512 capability detection during configuration
- Flexible flag testing to find optimal compiler settings
- Cross-platform support (Linux, Windows, macOS)
- Runtime detection when possible

#### Usage
```bash
# Enable AVX-512 if available
cmake -DLIBSTATS_ENABLE_AVX512=ON

# Force AVX-512 compilation (for testing)
cmake -DLIBSTATS_FORCE_AVX512=ON

# Test compilation but skip runtime checks
cmake -DLIBSTATS_TEST_AVX512_COMPILATION=ON
```

## Setting Up AVX-512 Testing

### Step 1: Enable Compilation Testing (Done)
The compilation verification workflow is already created and will run automatically.

### Step 2: Add CMake Support
```bash
# Add to your main CMakeLists.txt
include(cmake/DetectAVX512.cmake)
```

### Step 3: Set Up Self-Hosted Runner (Optional)
For complete testing, follow the self-hosted runner setup above.

## AVX-512 Hardware Availability

### Cloud Providers with AVX-512
- **AWS**: C5n, M5n, R5n, C6i, M6i, R6i instances
- **Azure**: Dv4, Ev4, Dv5, Ev5 series
- **GCP**: C2 machine family (Cascade Lake)

### GitHub Codespaces
Some Codespaces instances may have AVX-512, but it's not guaranteed.

### Docker Solutions
```dockerfile
# Example Dockerfile for AVX-512 testing
FROM ubuntu:22.04
RUN apt-get update && apt-get install -y \
    gcc-12 g++-12 cmake git \
    && rm -rf /var/lib/apt/lists/*

# Test AVX-512 availability
RUN if grep -q avx512f /proc/cpuinfo; then \
        echo "AVX-512 available"; \
    else \
        echo "No AVX-512 support"; \
    fi
```

## Monitoring AVX-512 Test Results

### Workflow Status
- Check `.github/workflows/avx512-compilation.yml` results
- Look for "AVX-512 Compilation Test" in Actions tab

### Key Indicators
- ✅ **Compilation successful**: AVX-512 code compiles without errors
- ✅ **Object files contain AVX-512**: `objdump` shows AVX-512 instructions
- ✅ **Headers parse correctly**: No preprocessor errors
- ⚠️ **Runtime testing skipped**: No AVX-512 hardware available

### Failure Investigation
If AVX-512 compilation fails:
1. Check compiler version (GCC 9+, Clang 10+ recommended)
2. Verify intrinsic header availability
3. Check for conflicting compiler flags
4. Review CMake configuration output

## Future Enhancements

### Potential Additions
1. **Emulation testing**: Use Intel SDE (Software Development Emulator)
2. **Cross-compilation**: Test for different AVX-512 variants
3. **Performance regression**: Compare AVX-512 vs AVX2 performance
4. **Feature-specific testing**: Test AVX-512DQ, AVX-512BW, etc. separately

### Integration with CI/CD
```yaml
# Example of conditional AVX-512 testing
- name: Extended SIMD Testing
  if: contains(github.event.pull_request.labels.*.name, 'simd-testing')
  run: |
    # Run comprehensive SIMD tests including AVX-512 compilation
    cmake -DLIBSTATS_ENABLE_ALL_SIMD_TESTS=ON
    cmake --build build --parallel
    ctest -L simd
```

This comprehensive approach ensures that while we may not have runtime AVX-512 testing on every commit, we maintain confidence that our AVX-512 code is correct and will work when deployed to appropriate hardware.

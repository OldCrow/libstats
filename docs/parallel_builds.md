# Parallel Build Configuration

libstats now includes automatic parallel job detection and configuration to optimize build times across different platforms.

## Automatic Parallel Job Detection

The CMake build system automatically detects the number of CPU cores and configures parallel builds:

### CMake Configuration
- **CPU Detection**: Uses `include(ProcessorCount)` and `ProcessorCount(CPU_COUNT)` 
- **Cross-Platform**: Works on macOS (`sysctl -n hw.ncpu`), Linux (`nproc`), and Windows
- **Fallback**: Defaults to 4 parallel jobs if detection fails
- **Environment Variables**: Sets `MAKEFLAGS=-j{CPU_COUNT}` for make-based builds
- **CMake Integration**: Sets `CMAKE_BUILD_PARALLEL_LEVEL={CPU_COUNT}` for `cmake --build`

### Messages During Configuration
```
-- Detected 8 CPU cores - enabling parallel builds
-- Set MAKEFLAGS=-j8 for make-based build  
-- Set CMAKE_BUILD_PARALLEL_LEVEL=8 for cmake --build
```

## Build Script Usage

A comprehensive build script is provided at `scripts/build.sh` with automatic parallel job detection:

### Basic Usage
```bash
# Build with auto-detected parallel jobs
./scripts/build.sh

# Build specific configuration with parallel jobs
./scripts/build.sh Release

# Clean build with tests
./scripts/build.sh -c -t Release
```

### Advanced Options
```bash
# Override parallel job count
./scripts/build.sh -j 16 Release

# Configure only (no build)
./scripts/build.sh --configure-only Dev

# Build only (skip configure)
./scripts/build.sh --build-only

# Verbose output with parallel builds
./scripts/build.sh -v Release
```

### Build Types Supported
- `Debug` - Debug build with full debug info
- `Release` - Optimized release build  
- `Dev` - Development build (default) with light optimization + debug info
- `ClangStrict` - Clang with strict warnings as errors
- `ClangWarn` - Clang with strict warnings (not errors)
- `GCCStrict` - GCC with strict warnings as errors  
- `GCCWarn` - GCC with strict warnings (not errors)
- `MSVCStrict` - MSVC with strict warnings as errors
- `MSVCWarn` - MSVC with strict warnings (not errors)

## Manual Build Commands

### Using CMake directly
```bash
# Configure with automatic parallel detection
cmake -DCMAKE_BUILD_TYPE=Release ..

# Build with detected parallel jobs
cmake --build . --parallel  # Uses CMAKE_BUILD_PARALLEL_LEVEL

# Or specify job count manually
cmake --build . --parallel 8
```

### Using Make directly  
```bash
# Configure sets MAKEFLAGS=-j{CPU_COUNT}
cmake -DCMAKE_BUILD_TYPE=Release ..

# Make automatically uses MAKEFLAGS
make

# Or specify manually 
make -j8
```

## Performance Benefits

The automatic parallel build configuration provides significant performance improvements:

- **Utilizes all CPU cores** for compilation
- **Reduces build times** by 4-8x on multi-core systems
- **Cross-platform compatibility** with automatic detection
- **Zero configuration required** for optimal performance

## Platform-Specific Details

### macOS  
- Detection: `sysctl -n hw.ncpu`
- AppleClang compiler with NEON SIMD acceleration
- Homebrew LLVM support when available

### Linux
- Detection: `nproc` command or `/proc/cpuinfo` parsing  
- GCC/Clang compiler support
- Full SIMD detection (SSE2, AVX, AVX2, AVX-512)

### Windows
- Detection: `ProcessorCount` CMake module
- MSVC and ClangCL compiler support
- Visual Studio integration

## Troubleshooting

### Build Script Issues
```bash
# Check if script is executable
chmod +x scripts/build.sh

# Run with explicit shell
bash scripts/build.sh Release
```

### CPU Detection Issues
- Check CMake output during configuration
- Manually override with `-j` flag in build script
- Set `CMAKE_BUILD_PARALLEL_LEVEL` environment variable

### Memory Constraints
If you experience out-of-memory issues during parallel builds:
```bash
# Reduce parallel jobs
./scripts/build.sh -j 4 Release

# Or use sequential build
./scripts/build.sh -j 1 Release
```

## Example Output

Successful parallel build configuration:
```
[INFO] libstats Build Script
[INFO] Project Root: /path/to/libstats  
[INFO] Build Type: Release
[INFO] Detected 8 CPU cores using sysctl (macOS/BSD)
[INFO] Building with 8 parallel jobs...
-- Detected 8 CPU cores - enabling parallel builds
-- Set MAKEFLAGS=-j8 for make-based build
-- Set CMAKE_BUILD_PARALLEL_LEVEL=8 for cmake --build
```

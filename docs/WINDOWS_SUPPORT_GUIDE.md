# Windows Support Branch

This branch contains Windows-specific enhancements to libstats for improved compatibility with Windows development environments, particularly Visual Studio 2022 and MSVC.

## Changes Made

### 1. CMakeLists.txt Enhancements

- **MSVC Detection**: Updated compiler detection logic to properly handle MSVC and comprehensive SIMD detection
- **Windows Definitions**: Added proper Windows-specific preprocessor definitions (`-DNOMINMAX`, `-D_USE_MATH_DEFINES`)
- **Symbol Export**: Enabled `CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS` for simplified DLL builds
- **Compiler Flags**: Added appropriate MSVC compiler flags (`/W4 /O2`)

### 2. SIMDDetection.cmake Comprehensive Windows Support

- **MSVC/Clang-cl Detection**: Added logic to detect MSVC and Clang-cl on Windows
- **Windows SIMD Flags**: Implemented Windows-specific SIMD compiler flags:
  - `/arch:SSE2` for SSE2 support
  - `/arch:AVX` for AVX support
  - `/arch:AVX2` for AVX2 support
  - `/arch:AVX512` for AVX-512 support
- **Runtime Testing**: Enhanced runtime CPU feature testing to work with both GCC/Clang and MSVC compilers
- **Cross-Platform Compatibility**: Maintained full backward compatibility with Unix/Linux/macOS builds

### 3. SIMD Compilation Configuration

- **Platform-Specific Flags**: Updated `configure_simd_target` function to use appropriate compilation flags based on compiler:
  - MSVC/Clang-cl on Windows: `/arch:*` flags
  - GCC/Clang on Unix: `-m*` flags
- **Conditional Compilation**: All Windows-specific code is wrapped in proper conditional compilation checks

## Development Environment Setup

### Recommended Windows Setup

1. **Visual Studio 2022 Community Edition** (free)
   - Install with "Desktop development with C++" workload
   - Include CMake tools for C++
   - Include Git for Windows

2. **Alternative: LLVM/Clang for Windows**
   - Download LLVM Windows binaries
   - Or install via Chocolatey: `choco install llvm cmake git`

3. **VS Code Option**
   - Visual Studio Code with C/C++ extension pack
   - CMake Tools extension
   - Works well with both MSVC and Clang toolchains

### Build Instructions for Windows

```powershell
# Clone the repository
git clone https://github.com/your-repo/libstats.git
cd libstats

# Switch to windows-support branch
git checkout windows-support

# Create build directory
mkdir build
cd build

# Configure for Visual Studio 2022, 64-bit
cmake .. -G "Visual Studio 17 2022" -A x64

# Build Debug configuration
cmake --build . --config Debug

# Build Release configuration
cmake --build . --config Release

# Run tests
ctest -C Release --output-on-failure
```

## Hardware Optimization

The enhanced SIMD detection will automatically detect and optimize for:

- **Intel i7 CPUs**: Full SSE2/AVX/AVX2 support with proper MSVC flag detection
- **Modern Intel/AMD**: AVX-512 support where available
- **Cross-compilation**: Environment variable overrides for build systems

## Compatibility

- **Maintains Full Compatibility**: All existing Unix/Linux/macOS builds continue to work unchanged
- **Conditional Compilation**: Windows-specific code only activates on Windows platforms
- **Fallback Support**: Robust fallback mechanisms for unsupported SIMD features

## Testing

The Windows support has been designed with comprehensive testing in mind:

- **SIMD Feature Detection**: Automatic runtime testing of CPU capabilities
- **Compiler Flag Validation**: Verification that SIMD flags are supported by the compiler
- **Cross-Platform CI**: Ready for Windows CI/CD integration

## Integration Notes

### Merging Back to Main

When ready to merge back to main:

1. All changes use conditional compilation (`#ifdef WIN32`, `if(MSVC)`)
2. No existing functionality is modified, only extended
3. Unix/Linux/macOS builds remain completely unchanged
4. Windows builds gain full SIMD optimization support

### Future Enhancements

Potential areas for future Windows-specific improvements:

1. **CPU Topology Detection**: Windows-specific CPU core/thread detection
2. **Memory Management**: Windows-specific memory allocation optimizations
3. **Thread Affinity**: Windows thread affinity APIs for performance
4. **Visual Studio Integration**: Project templates and debugging support

## Known Limitations

- **AVX-512 Support**: Limited by both hardware availability and compiler support
- **Runtime Detection**: Some older Windows systems may have limited CPU feature detection
- **Cross-Compilation**: Manual environment variable setup may be needed for complex build scenarios

## Contact and Support

This Windows support branch maintains full compatibility while adding comprehensive Windows development environment support. All changes are isolated using platform detection, ensuring safe integration with the main codebase.

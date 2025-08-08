# CMAKE_BUILD_TYPE Reference for libstats

This document provides a complete reference for all available CMAKE_BUILD_TYPE options in the libstats project, organized for cross-platform development and compiler testing.

## Quick Reference

| Build Type | Purpose | Default | Optimization | Debug Info | Warnings as Errors |
|------------|---------|---------|--------------|------------|-------------------|
| **Dev** | Daily development | ✅ **DEFAULT** | Light (-O1) | Yes | No |
| **Debug** | Full debugging | | None (-O0) | Yes | No |
| **Release** | Production builds | | Maximum (-O2/-O3) | No | No |
| **ClangStrict** | Clang error checking | | Standard | Standard | ✅ Yes |
| **ClangWarn** | Clang development | | Standard | Standard | No |
| **MSVCStrict** | MSVC error checking | | Standard | Standard | ✅ Yes |
| **MSVCWarn** | MSVC development | | Standard | Standard | No |
| **GCCStrict** | GCC error checking | | Standard | Standard | ✅ Yes |
| **GCCWarn** | GCC development | | Standard | Standard | No |

## Standard Build Types (Cross-platform)

These build types work across all compilers and platforms, automatically selecting appropriate flags based on the detected compiler.

### Dev (Default)
**Purpose**: Optimal for daily development work - provides good debugging capabilities while maintaining reasonable performance.

- **MSVC**: `/W3 /O1 /Zi /MD`
- **Clang**: `-O1 -g -Wall -Wextra`  
- **GCC**: `-O1 -g -Wall -Wextra`

**Usage**: `cmake ..` (no flag needed - this is the default)

### Release
**Purpose**: Maximum performance for production builds.

- **MSVC**: `/W3 /O2 /DNDEBUG`
- **Clang**: `-O3 -DNDEBUG -Wall`
- **GCC**: `-O3 -DNDEBUG -Wall`

**Usage**: `cmake -DCMAKE_BUILD_TYPE=Release ..`

### Debug
**Purpose**: Full debugging with extensive runtime checks.

- **MSVC**: `/W3 /Od /Zi /RTC1 /MDd`
- **Clang**: `-O0 -g -Wall -Wextra`
- **GCC**: `-O0 -g -Wall -Wextra -fstack-protector-strong`

**Usage**: `cmake -DCMAKE_BUILD_TYPE=Debug ..`

## Compiler-Specific Build Types

These build types enable you to test your code against different compiler warning standards and catch compiler-specific issues.

### Clang Build Types

#### ClangStrict
**Purpose**: Strict Clang-specific error checking - warnings become errors.

**Flags**: 
```
-Wall -Wextra -Werror -pedantic
-Wconversion -Wfloat-conversion -Wimplicit-int-conversion -Wshorten-64-to-32
-Wdouble-promotion -Wsign-conversion
-Wshadow -Wunused -Wcast-align -Wcast-qual
-Wold-style-cast -Woverloaded-virtual -Wextra-semi -Wmissing-declarations
-Wloop-analysis -Wlogical-op-parentheses -Wbool-conversion
-Wint-conversion -Wnull-dereference -Wuninitialized
-Wconditional-uninitialized -Wstring-conversion
-Wno-unused-parameter -Wno-padded -Wno-c++98-compat -Wno-c++98-compat-pedantic
```

**Usage**: `cmake -DCMAKE_BUILD_TYPE=ClangStrict ..`

#### ClangWarn
**Purpose**: Same warnings as ClangStrict but as warnings (no `-Werror`), plus `-Wno-deprecated-declarations`.

**Usage**: `cmake -DCMAKE_BUILD_TYPE=ClangWarn ..`

### MSVC Build Types

#### MSVCStrict
**Purpose**: Strict MSVC-specific error checking - warnings become errors.

**Flags**:
```
/W4 /WX /permissive-
/w14242 /w14254 /w14263 /w14265 /w14287 /we4289 /w14296 /w14311 
/w14545 /w14546 /w14547 /w14549 /w14555 /w14619 /w14640 
/w14826 /w14905 /w14906 /w14928
```

**Usage**: `cmake -DCMAKE_BUILD_TYPE=MSVCStrict ..`

#### MSVCWarn
**Purpose**: Same warnings as MSVCStrict but as warnings (no `/WX`).

**Usage**: `cmake -DCMAKE_BUILD_TYPE=MSVCWarn ..`

### GCC Build Types

#### GCCStrict
**Purpose**: Strict GCC-specific error checking - warnings become errors.

**Flags**:
```
-Wall -Wextra -Werror -pedantic
-Wconversion -Wfloat-conversion -Wdouble-promotion -Wsign-conversion -Wold-style-cast
-Wlogical-op -Wduplicated-cond -Wduplicated-branches
-Wrestrict -Wnull-dereference -Wjump-misses-init
-Wtrampolines -Wunsafe-loop-optimizations -Wvector-operation-performance
-Wsuggest-override -Wsuggest-final-types -Wsuggest-final-methods
-Wcast-align=strict -Wstrict-overflow=2
-Wformat=2 -Wformat-overflow=2 -Wformat-truncation=2
-Wshift-overflow=2 -Wstringop-overflow=4
-Wno-inline -Wno-padded -Wno-unused-parameter -Wno-sign-conversion
```

**Usage**: `cmake -DCMAKE_BUILD_TYPE=GCCStrict ..`

#### GCCWarn
**Purpose**: Same warnings as GCCStrict but as warnings (no `-Werror`).

**Usage**: `cmake -DCMAKE_BUILD_TYPE=GCCWarn ..`

## Cross-Compiler Compatibility Testing

For testing MSVC compatibility while using Clang, you can also use:

- `cmake -DCMAKE_BUILD_TYPE=MSVCStrict ..` (with Clang compiler)
- `cmake -DCMAKE_BUILD_TYPE=MSVCWarn ..` (with Clang compiler)

This applies MSVC-like warning flags to Clang for compatibility testing.

## Development Workflow Recommendations

1. **Daily Development**: Use `Dev` (default) - fast compilation with good debugging
2. **Before Committing**: Test with `ClangWarn` or `GCCWarn` to catch issues
3. **CI/CD Pipeline**: Use `ClangStrict`, `MSVCStrict`, `GCCStrict` to ensure code quality
4. **Performance Testing**: Use `Release` for benchmarking
5. **Deep Debugging**: Use `Debug` when you need maximum debugging capabilities

## Examples

```bash
# Default development build
mkdir build && cd build && cmake ..

# Test with strict Clang checking
mkdir build_clang_strict && cd build_clang_strict
cmake -DCMAKE_BUILD_TYPE=ClangStrict ..

# Release build for performance testing  
mkdir build_release && cd build_release
cmake -DCMAKE_BUILD_TYPE=Release ..

# Test MSVC compatibility (using Clang)
mkdir build_msvc_test && cd build_msvc_test  
cmake -DCMAKE_BUILD_TYPE=MSVCStrict ..
```

## Notes

- **SIMD Support**: All build types include appropriate SIMD optimizations
- **Platform Detection**: Flags are automatically selected based on detected compiler
- **TBB Integration**: Threading Building Blocks support when available
- **Debug Symbols**: Dev and Debug builds include debug information for optimal debugging experience
- **Stack Protection**: GCC Debug builds include `-fstack-protector-strong` for enhanced security

The build system is designed to support your cross-platform development workflow while maintaining code quality across different compiler ecosystems.

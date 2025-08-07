# Using C++20 with Homebrew LLVM on Older MacOS

This guide provides a comprehensive solution for configuring C++20 compilation and linking using Homebrew LLVM on older MacOS systems where Apple's default toolchain may not support modern C++ standards.

## Problem Context

Older MacOS systems often come with outdated versions of Clang that don't fully support C++20 features. Additionally, linking against the system's standard library can cause issues with modern C++ features like `std::function`, concepts, and ranges.

## Solution Overview

Use Homebrew LLVM with proper configuration to get full C++20 support with modern libc++.

## Prerequisites

- MacOS system (tested on macOS 10.15+)
- Homebrew package manager installed
- CMake 3.15 or later

## Step-by-Step Configuration

### 1. Install Homebrew LLVM

```bash
brew install llvm
```

This installs LLVM/Clang with full C++20 support, typically in `/usr/local/opt/llvm/`.

### 2. Set Environment Variables

Configure your shell environment to use Homebrew LLVM:

```bash
export CPPFLAGS="-I/usr/local/opt/llvm/include"
export LDFLAGS="-L/usr/local/opt/llvm/lib"
```

Add these to your shell profile (`.zshrc`, `.bashrc`, etc.) for persistence.

### 3. CMake Configuration

Configure your `CMakeLists.txt` with proper Homebrew LLVM settings:

```cmake
cmake_minimum_required(VERSION 3.15)

# Use Homebrew LLVM for C++20 support
set(CMAKE_C_COMPILER "/usr/local/opt/llvm/bin/clang")
set(CMAKE_CXX_COMPILER "/usr/local/opt/llvm/bin/clang++")

# Configure LLVM paths
set(LLVM_ROOT "/usr/local/opt/llvm")
set(CMAKE_PREFIX_PATH "${LLVM_ROOT}")

project(your_project VERSION 1.0.0 LANGUAGES CXX)

# C++20 requirement
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Homebrew LLVM specific configuration
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L${LLVM_ROOT}/lib/c++ -Wl,-rpath,${LLVM_ROOT}/lib/c++")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -L${LLVM_ROOT}/lib/c++ -Wl,-rpath,${LLVM_ROOT}/lib/c++")

# Include directories for LLVM
include_directories(${LLVM_ROOT}/include/c++/v1)
include_directories(${LLVM_ROOT}/include)

# Standard compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -O3 -march=native)
endif()
```

### 4. Manual Compilation Template

For direct command-line compilation:

```bash
/usr/local/opt/llvm/bin/clang++ -std=c++20 -stdlib=libc++ \
  -I/usr/local/opt/llvm/include/c++/v1 \
  -I./include \
  -L/usr/local/opt/llvm/lib/c++ \
  -L./build \
  -Wl,-rpath,/usr/local/opt/llvm/lib/c++ \
  -llibstats_compiled \
  source.cpp -o output_binary
```

### 5. Build Process

With CMake:

```bash
# Clean build
rm -rf build && mkdir build && cd build

# Configure with Release build
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
make -j$(nproc)
```

## Verification

### Check C++20 Support

Create a test file to verify C++20 features work:

```cpp
#include <iostream>
#include <concepts>
#include <ranges>
#include <vector>

template<typename T>
concept Numeric = std::integral<T> || std::floating_point<T>;

template<Numeric T>
T add(T a, T b) { return a + b; }

int main() {
    std::cout << "C++ Standard: " << __cplusplus << std::endl;
    
    // Test concepts
    auto result = add(5, 10);
    std::cout << "Concepts work: " << result << std::endl;
    
    // Test ranges
    std::vector<int> data = {1, 2, 3, 4, 5};
    auto transformed = data | std::views::transform([](int x) { return x * 2; });
    
    std::cout << "Ranges work: ";
    for (auto x : transformed) {
        std::cout << x << " ";
    }
    std::cout << std::endl;
    
    return 0;
}
```

Compile and run:

```bash
/usr/local/opt/llvm/bin/clang++ -std=c++20 -stdlib=libc++ \
  -I/usr/local/opt/llvm/include/c++/v1 \
  test.cpp -o test && ./test
```

Expected output:
```
C++ Standard: 202002
Concepts work: 15
Ranges work: 2 4 6 8 10
```

## Key Configuration Points

### Critical Flags

1. **`-stdlib=libc++`**: Use LLVM's modern C++ standard library
2. **`-I/usr/local/opt/llvm/include/c++/v1`**: Include modern C++ headers
3. **`-L/usr/local/opt/llvm/lib/c++`**: Link against LLVM's libc++
4. **`-Wl,-rpath,/usr/local/opt/llvm/lib/c++`**: Runtime library path

### Common Issues Solved

- **`std::bad_function_call` undefined symbols**: Fixed by proper libc++ linking
- **Missing C++20 features**: Resolved with Homebrew LLVM vs system Clang
- **Threading + std::function issues**: Solved with correct standard library
- **Ranges/concepts compilation errors**: Fixed with modern libc++

## Tested Features

✅ **C++20 Concepts**: Template constraints and requirements  
✅ **C++20 Ranges**: Views, transformations, and algorithms  
✅ **std::function + Threading**: No more linking errors  
✅ **Modern STL**: Full C++20 standard library support  
✅ **Template Lambdas**: Advanced lambda template features  

## Environment Details

- **Tested on**: macOS 10.15+ with Homebrew LLVM 20.1.7
- **CMake**: 3.15+
- **Build System**: Make, Ninja compatible
- **Library Types**: Static and shared libraries supported

## Local Library Linking on macOS

### Dynamic vs Static Linking for Local Projects

When working with locally built libraries that are **not installed** to standard system paths (like `/usr/local/lib` or `/usr/lib`), macOS has specific requirements for dynamic linking that often cause runtime errors.

#### The Problem

If your project builds a dynamic library (`.dylib`) in a local build directory, attempting to link against it and run the resulting executable will fail with:

```
dyld: Library not loaded: @rpath/libYourProject.dylib
  Referenced from: /path/to/your/executable
  Reason: image not found
```

This happens because macOS's dynamic linker (`dyld`) cannot find the library at runtime - it's not in the standard system library directories and the executable doesn't know where to look for it.

#### Solution 1: Static Linking (Recommended for Local Development)

The simplest solution is to link against the static library (`.a` file) instead:

```bash
/usr/local/opt/llvm/bin/clang++ -std=c++20 -stdlib=libc++ \
  -I/usr/local/opt/llvm/include/c++/v1 \
  -I./include \
  -L/usr/local/opt/llvm/lib/c++ \
  -L./build \
  -Wl,-rpath,/usr/local/opt/llvm/lib/c++ \
  source.cpp -o output_binary ./build/libYourProject.a
```

Or using the `-l` flag:

```bash
/usr/local/opt/llvm/bin/clang++ -std=c++20 -stdlib=libc++ \
  -I/usr/local/opt/llvm/include/c++/v1 \
  -I./include \
  -L/usr/local/opt/llvm/lib/c++ \
  -L./build \
  -Wl,-rpath,/usr/local/opt/llvm/lib/c++ \
  -static \
  source.cpp -o output_binary -lYourProject
```

#### Solution 2: Dynamic Linking with Local Library Path

If you must use dynamic linking, you need to tell the linker where to find the library at runtime by adding the local build directory to the rpath:

```bash
/usr/local/opt/llvm/bin/clang++ -std=c++20 -stdlib=libc++ \
  -I/usr/local/opt/llvm/include/c++/v1 \
  -I./include \
  -L/usr/local/opt/llvm/lib/c++ \
  -L./build \
  -Wl,-rpath,/usr/local/opt/llvm/lib/c++ \
  -Wl,-rpath,./build \
  source.cpp -o output_binary -lYourProject
```

**Note:** The `-Wl,-rpath,./build` tells the dynamic linker to also search in the `./build` directory for libraries at runtime.

#### Solution 3: Environment Variable (Temporary)

For quick testing, you can set the library path at runtime:

```bash
DYLD_LIBRARY_PATH=./build ./output_binary
```

**Warning:** This method is not recommended for production and may not work with System Integrity Protection (SIP) enabled on newer macOS versions.

### Best Practice Recommendations

1. **For development/testing**: Use static linking (`.a` files) - it's simpler and avoids path issues
2. **For distributed applications**: Use dynamic linking with proper installation to standard paths
3. **For local libraries in build directories**: Always use static linking or add the build directory to rpath

## Troubleshooting

### Compilation Errors
- Ensure `/usr/local/opt/llvm/bin/clang++` is used, not system clang++
- Verify `-stdlib=libc++` flag is present
- Check include paths point to LLVM headers

### Linking Errors
- Confirm `-L/usr/local/opt/llvm/lib/c++` is in linker flags
- Verify rpath is set: `-Wl,-rpath,/usr/local/opt/llvm/lib/c++`
- Check that LLVM libraries exist: `ls /usr/local/opt/llvm/lib/c++/`
- **For local libraries**: Use static linking or add build directory to rpath

### Runtime Issues
- Ensure rpath is correctly set for dynamic linking
- Verify LLVM libraries are accessible: `otool -L your_binary`
- **Library not found errors**: Check if using local dynamic libraries - switch to static linking

This configuration provides a robust foundation for C++20 development on older MacOS systems using Homebrew LLVM.

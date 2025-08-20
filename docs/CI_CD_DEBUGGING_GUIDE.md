# CI/CD Debugging and Issue Resolution Guide

## 1. Identifying CI/CD Issues

### Viewing CI Results

#### GitHub Actions Web Interface
1. Go to your repository on GitHub
2. Click on the "Actions" tab
3. Select the workflow run to inspect
4. Click on failed jobs to see detailed logs

#### Command Line (using GitHub CLI)
```bash
# Install GitHub CLI if needed
brew install gh

# List recent workflow runs
gh run list --limit 10

# View a specific run
gh run view <run-id>

# Watch a run in progress
gh run watch

# Download logs for debugging
gh run download <run-id>
```

### Common Issue Categories

1. **Build Failures**
   - Compiler errors
   - Linker errors
   - Missing dependencies
   - Platform-specific issues

2. **Test Failures**
   - Unit test failures
   - Timeout issues
   - Platform-specific test failures

3. **Code Quality Issues**
   - Formatting violations
   - Static analysis warnings
   - Code coverage drops

## 2. Local Reproduction of CI Issues

### Script to Simulate CI Environment Locally

Create `scripts/ci-local.sh`:
```bash
#!/bin/bash
# Simulate CI environment locally

# Parse arguments
COMPILER=${1:-gcc}
BUILD_TYPE=${2:-Debug}
RUN_LINT=${3:-yes}

echo "Simulating CI with: $COMPILER $BUILD_TYPE"

# Clean build directory
rm -rf build-ci
mkdir build-ci

# Set compiler
case $COMPILER in
    gcc|gcc-11|gcc-12)
        export CC=gcc
        export CXX=g++
        ;;
    clang|clang-14|clang-15)
        export CC=clang
        export CXX=clang++
        ;;
esac

# Configure with CI-like settings
cmake -B build-ci \
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
    -DCMAKE_CXX_STANDARD=20 \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DCMAKE_CXX_FLAGS="-Wall -Wextra -Wpedantic -Werror"

# Build
cmake --build build-ci --parallel

# Run tests
cd build-ci && ctest --output-on-failure
cd ..

# Run linting if requested
if [ "$RUN_LINT" = "yes" ]; then
    ./scripts/lint.sh
fi
```

## 3. Fixing Common Issues

### Build Failures

#### Compiler Errors
```bash
# Get detailed error output
cmake --build build --verbose

# For template errors, use better diagnostics
cmake -B build -DCMAKE_CXX_FLAGS="-ftemplate-backtrace-limit=0"
```

#### Platform-Specific Issues
```bash
# Test with different standards
cmake -B build -DCMAKE_CXX_STANDARD=17  # fallback
cmake -B build -DCMAKE_CXX_STANDARD=20  # current
cmake -B build -DCMAKE_CXX_STANDARD=23  # future
```

### Test Failures

#### Debugging Test Failures
```bash
# Run specific test with verbose output
ctest -R test_name -VV

# Run with debugger
gdb ./build/tests/test_executable
lldb ./build/tests/test_executable

# Run with sanitizers
cmake -B build-asan -DCMAKE_CXX_FLAGS="-fsanitize=address"
cmake --build build-asan
./build-asan/tests/test_executable
```

### Code Quality Issues

#### Formatting Issues
```bash
# See what would change
find src include -name "*.cpp" -o -name "*.h" | \
    xargs clang-format --dry-run --Werror

# Fix automatically
./scripts/format.sh

# Check specific file
clang-format --dry-run --Werror src/specific_file.cpp
```

#### Static Analysis Warnings
```bash
# Run clang-tidy on specific file with fix suggestions
clang-tidy src/file.cpp -p build --fix

# See all checks
clang-tidy --list-checks

# Run specific check category
clang-tidy src/file.cpp -p build \
    --checks="-*,performance-*"
```

## 4. Updating CMakeLists.txt for Better Local Checking

### Enhanced Warning Configuration

Add to your main `CMakeLists.txt`:

```cmake
# Compiler warning flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(WARNING_FLAGS
        -Wall
        -Wextra
        -Wpedantic
        -Wshadow
        -Wnon-virtual-dtor
        -Wold-style-cast
        -Wcast-align
        -Wunused
        -Woverloaded-virtual
        -Wconversion
        -Wsign-conversion
        -Wnull-dereference
        -Wdouble-promotion
        -Wformat=2
    )

    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        list(APPEND WARNING_FLAGS
            -Wmisleading-indentation
            -Wduplicated-cond
            -Wduplicated-branches
            -Wlogical-op
            -Wuseless-cast
        )
    endif()

    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        list(APPEND WARNING_FLAGS
            -Wno-gnu-zero-variadic-macro-arguments
            -Wno-c++98-compat
            -Wno-c++98-compat-pedantic
        )
    endif()
elseif(MSVC)
    set(WARNING_FLAGS
        /W4
        /WX
        /permissive-
        /w14242 /w14254 /w14263
        /w14265 /w14287 /we4289
        /w14296 /w14311 /w14545
        /w14546 /w14547 /w14549
        /w14555 /w14619 /w14640
        /w14826 /w14905 /w14906
        /w14928
    )
endif()

# Option to treat warnings as errors
option(LIBSTATS_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)
if(LIBSTATS_WARNINGS_AS_ERRORS)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        list(APPEND WARNING_FLAGS -Werror)
    elseif(MSVC)
        list(APPEND WARNING_FLAGS /WX)
    endif()
endif()

# Apply to targets
target_compile_options(libstats PRIVATE ${WARNING_FLAGS})
```

### Build Configurations for CI Matching

```cmake
# CI configuration preset
if(LIBSTATS_CI_BUILD)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g3")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")

    # Enable all warnings in CI
    set(LIBSTATS_WARNINGS_AS_ERRORS ON)

    # Enable sanitizers in debug builds
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        option(LIBSTATS_ENABLE_ASAN "Enable Address Sanitizer" ON)
        option(LIBSTATS_ENABLE_UBSAN "Enable Undefined Behavior Sanitizer" ON)
    endif()
endif()
```

## 5. Automated Issue Detection Scripts

### Pre-commit Hook

Create `.git/hooks/pre-commit`:
```bash
#!/bin/bash
# Pre-commit hook to catch issues before CI

echo "Running pre-commit checks..."

# Check formatting
if ! ./scripts/lint.sh; then
    echo "❌ Linting failed. Fix issues before committing."
    exit 1
fi

# Quick build test
if ! cmake --build build --target libstats; then
    echo "❌ Build failed. Fix compilation errors."
    exit 1
fi

echo "✅ Pre-commit checks passed!"
```

### CI Issue Scanner

Create `scripts/scan-ci-issues.sh`:
```bash
#!/bin/bash
# Scan for common CI issues in codebase

echo "Scanning for potential CI issues..."

# Check for non-portable code
echo "Checking for platform-specific code..."
grep -r "ifdef.*WIN32\|ifdef.*__linux__\|ifdef.*__APPLE__" src include

# Check for missing includes
echo "Checking for potentially missing includes..."
grep -r "std::" src include | grep -v "#include"

# Check for large files that might timeout
echo "Checking for large files..."
find . -type f -name "*.cpp" -o -name "*.h" | xargs wc -l | sort -rn | head -10

# Check for hardcoded paths
echo "Checking for hardcoded paths..."
grep -r "/usr/\|/home/\|C:\\\\" src include tests

echo "Scan complete!"
```

## 6. Monitoring and Fixing Workflow

### Daily Workflow
1. Check CI status on main branch
2. Review any new warnings or failures
3. Create fix branches for issues
4. Test fixes locally with CI simulation
5. Submit PRs with fixes

### Weekly Tasks
1. Review CI performance metrics
2. Update warning suppressions if needed
3. Upgrade tool versions if available
4. Review and merge warning fixes

### Per-Release Tasks
1. Enable stricter warning levels
2. Remove warning suppressions
3. Update compiler version requirements
4. Performance baseline updates

## 7. Quick Reference

### Fix Commands by Issue Type

| Issue Type | Quick Fix Command |
|------------|------------------|
| Format | `./scripts/format.sh` |
| Build | `cmake --build build --verbose` |
| Test | `ctest -R failing_test -VV` |
| Tidy | `clang-tidy file.cpp -p build --fix` |
| Warning | See compiler output, fix in source |

### CI Environment Variables

Set these locally to match CI:
```bash
export CXXFLAGS="-Wall -Wextra -Wpedantic"
export CMAKE_BUILD_TYPE=Release
export CMAKE_CXX_STANDARD=20
```

## 8. Gradual Warning Enforcement

### Phase 1: Baseline (Current)
- Warnings reported but not failing
- Collect data on warning types

### Phase 2: Fix Critical (1-2 weeks)
- Fix all error-prone warnings
- Enable `-Werror` for critical warnings

### Phase 3: Full Enforcement (Before v1.0)
- Fix all warnings
- Enable `-Werror` globally
- Add to PR requirements

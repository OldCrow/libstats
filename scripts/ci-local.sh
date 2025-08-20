#!/bin/bash
# Simulate CI environment locally for libstats
# Usage: ./scripts/ci-local.sh [compiler] [build_type] [run_lint]
#   compiler: gcc, clang (default: gcc)
#   build_type: Debug, Release (default: Debug)
#   run_lint: yes, no (default: yes)

set -e

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Parse arguments
COMPILER=${1:-gcc}
BUILD_TYPE=${2:-Debug}
RUN_LINT=${3:-yes}

echo -e "${GREEN}===================================${NC}"
echo -e "${GREEN}LibStats CI Environment Simulation${NC}"
echo -e "${GREEN}===================================${NC}"
echo ""
echo "Configuration:"
echo "  Compiler: $COMPILER"
echo "  Build Type: $BUILD_TYPE"
echo "  Run Linting: $RUN_LINT"
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Clean build directory
echo -e "${YELLOW}Cleaning build directory...${NC}"
rm -rf build-ci
mkdir -p build-ci

# Set compiler based on what's available
echo -e "${YELLOW}Setting up compiler...${NC}"
case $COMPILER in
    gcc|gcc-11|gcc-12)
        # Find available GCC
        if command -v gcc-12 &> /dev/null; then
            export CC=gcc-12
            export CXX=g++-12
        elif command -v gcc-11 &> /dev/null; then
            export CC=gcc-11
            export CXX=g++-11
        elif command -v gcc &> /dev/null; then
            export CC=gcc
            export CXX=g++
        else
            echo -e "${RED}Error: GCC not found${NC}"
            exit 1
        fi
        ;;
    clang|clang-14|clang-15)
        # Find available Clang
        if command -v clang-15 &> /dev/null; then
            export CC=clang-15
            export CXX=clang++-15
        elif command -v clang-14 &> /dev/null; then
            export CC=clang-14
            export CXX=clang++-14
        elif command -v clang &> /dev/null; then
            export CC=clang
            export CXX=clang++
        else
            echo -e "${RED}Error: Clang not found${NC}"
            exit 1
        fi
        ;;
    *)
        echo -e "${RED}Unknown compiler: $COMPILER${NC}"
        exit 1
        ;;
esac

echo "Using: CC=$CC, CXX=$CXX"

# Configure with CI-like warning settings
echo ""
echo -e "${YELLOW}Configuring CMake with CI settings...${NC}"

# Base warning flags that both GCC and Clang support
WARNING_FLAGS="-Wall -Wextra -Wpedantic"

# Add more warnings for better coverage
if [[ "$CC" == *"gcc"* ]]; then
    WARNING_FLAGS="$WARNING_FLAGS -Wshadow -Wnon-virtual-dtor -Wold-style-cast"
    WARNING_FLAGS="$WARNING_FLAGS -Wcast-align -Wunused -Woverloaded-virtual"
    WARNING_FLAGS="$WARNING_FLAGS -Wconversion -Wsign-conversion -Wnull-dereference"
    WARNING_FLAGS="$WARNING_FLAGS -Wdouble-promotion -Wformat=2"
    WARNING_FLAGS="$WARNING_FLAGS -Wmisleading-indentation -Wduplicated-cond"
    WARNING_FLAGS="$WARNING_FLAGS -Wduplicated-branches -Wlogical-op"
elif [[ "$CC" == *"clang"* ]]; then
    WARNING_FLAGS="$WARNING_FLAGS -Wshadow -Wnon-virtual-dtor -Wold-style-cast"
    WARNING_FLAGS="$WARNING_FLAGS -Wcast-align -Wunused -Woverloaded-virtual"
    WARNING_FLAGS="$WARNING_FLAGS -Wconversion -Wsign-conversion -Wnull-dereference"
    WARNING_FLAGS="$WARNING_FLAGS -Wdouble-promotion -Wformat=2"
fi

# For now, don't treat warnings as errors to match current CI
# Uncomment this when ready to enforce: WARNING_FLAGS="$WARNING_FLAGS -Werror"

cmake -B build-ci \
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
    -DCMAKE_CXX_STANDARD=20 \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DCMAKE_CXX_FLAGS="$WARNING_FLAGS" \
    -DCMAKE_VERBOSE_MAKEFILE=ON

# Build
echo ""
echo -e "${YELLOW}Building project...${NC}"
if cmake --build build-ci --parallel 2>&1 | tee build-ci/build.log; then
    echo -e "${GREEN}✅ Build successful${NC}"
else
    echo -e "${RED}❌ Build failed${NC}"
    echo "Check build-ci/build.log for details"
    exit 1
fi

# Count warnings
WARNING_COUNT=$(grep -c "warning:" build-ci/build.log || true)
if [ "$WARNING_COUNT" -gt 0 ]; then
    echo -e "${YELLOW}⚠️  Found $WARNING_COUNT compiler warnings${NC}"
    echo "Review build-ci/build.log for details"
fi

# Run tests
echo ""
echo -e "${YELLOW}Running tests...${NC}"
cd build-ci
if ctest --output-on-failure --parallel; then
    echo -e "${GREEN}✅ All tests passed${NC}"
else
    echo -e "${RED}❌ Some tests failed${NC}"
    cd ..
    exit 1
fi
cd ..

# Run linting if requested
if [ "$RUN_LINT" = "yes" ]; then
    echo ""
    echo -e "${YELLOW}Running code quality checks...${NC}"

    # Check if clang-format is available
    if command -v clang-format &> /dev/null || command -v clang-format-15 &> /dev/null; then
        echo "Checking code formatting..."
        if ./scripts/lint.sh; then
            echo -e "${GREEN}✅ Linting passed${NC}"
        else
            echo -e "${YELLOW}⚠️  Linting issues found (not failing CI yet)${NC}"
        fi
    else
        echo -e "${YELLOW}Skipping format check (clang-format not found)${NC}"
    fi
fi

# Summary
echo ""
echo -e "${GREEN}===================================${NC}"
echo -e "${GREEN}CI Simulation Complete${NC}"
echo -e "${GREEN}===================================${NC}"
echo ""
echo "Summary:"
echo "  Build: ✅ Success"
echo "  Tests: ✅ Passed"
if [ "$WARNING_COUNT" -gt 0 ]; then
    echo "  Warnings: ⚠️  $WARNING_COUNT warnings found"
else
    echo "  Warnings: ✅ No warnings"
fi
if [ "$RUN_LINT" = "yes" ]; then
    echo "  Linting: ✅ Checked"
fi
echo ""
echo "Artifacts in build-ci/:"
echo "  - compile_commands.json (for clang-tidy)"
echo "  - build.log (build output with warnings)"
echo ""

#!/bin/bash

# libstats Build Script with Automatic Parallel Job Detection
# This script provides a convenient way to build libstats with optimal parallel compilation
# regardless of the underlying build system (make, ninja, etc.)

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to detect number of CPU cores
detect_cpu_cores() {
    local cpu_count=0
    local detection_method=""

    # Try different methods to detect CPU count
    if command -v nproc >/dev/null 2>&1; then
        cpu_count=$(nproc)
        detection_method="nproc (Linux)"
    elif command -v sysctl >/dev/null 2>&1; then
        if sysctl -n hw.ncpu >/dev/null 2>&1; then
            cpu_count=$(sysctl -n hw.ncpu)
            detection_method="sysctl (macOS/BSD)"
        fi
    elif [ -r /proc/cpuinfo ]; then
        cpu_count=$(grep -c ^processor /proc/cpuinfo)
        detection_method="/proc/cpuinfo"
    fi

    # Fallback if detection fails
    if [ "$cpu_count" -eq 0 ]; then
        cpu_count=4
        detection_method="fallback default"
    fi

    # Output info message to stderr so it doesn't interfere with return value
    if [ -n "$detection_method" ]; then
        print_info "Detected $cpu_count CPU cores using $detection_method" >&2
    fi

    echo $cpu_count
}

# Function to display usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS] [BUILD_TYPE]

Build libstats with automatic parallel job detection and optimization.

OPTIONS:
    -h, --help      Show this help message
    -j, --jobs N    Override auto-detected number of parallel jobs
    -c, --clean     Clean build directory before building
    -t, --tests     Run tests after building
    -v, --verbose   Enable verbose build output
    --configure-only Only configure, don't build
    --build-only    Only build (skip configure step)

BUILD_TYPE:
    Debug           Debug build with full debug info
    Release         Optimized release build
    Dev             Development build (default) - light optimization + debug info
    ClangStrict     Clang with strict warnings as errors
    ClangWarn       Clang with strict warnings (not errors)
    GCCStrict       GCC with strict warnings as errors
    GCCWarn         GCC with strict warnings (not errors)
    MSVCStrict      MSVC with strict warnings as errors
    MSVCWarn        MSVC with strict warnings (not errors)

EXAMPLES:
    $0                          # Build with Dev configuration and auto-detected parallel jobs
    $0 Release                  # Build Release configuration
    $0 -j 8 Debug               # Build Debug with 8 parallel jobs
    $0 -c -t Release            # Clean build, build Release, then run tests
    $0 --configure-only Dev     # Only configure for Dev build
    $0 --build-only             # Only build (don't reconfigure)

EOF
}

# Default values
BUILD_TYPE="Dev"
CLEAN_BUILD=false
RUN_TESTS=false
VERBOSE=false
CONFIGURE_ONLY=false
BUILD_ONLY=false
OVERRIDE_JOBS=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage
            exit 0
            ;;
        -j|--jobs)
            OVERRIDE_JOBS="$2"
            shift 2
            ;;
        -c|--clean)
            CLEAN_BUILD=true
            shift
            ;;
        -t|--tests)
            RUN_TESTS=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        --configure-only)
            CONFIGURE_ONLY=true
            shift
            ;;
        --build-only)
            BUILD_ONLY=true
            shift
            ;;
        Debug|Release|Dev|ClangStrict|ClangWarn|GCCStrict|GCCWarn|MSVCStrict|MSVCWarn)
            BUILD_TYPE="$1"
            shift
            ;;
        -*)
            print_error "Unknown option: $1"
            usage
            exit 1
            ;;
        *)
            # Assume it's a build type
            BUILD_TYPE="$1"
            shift
            ;;
    esac
done

# Get the directory of this script and find project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Use build type-specific directory if running from a specific build directory
CURRENT_DIR="$(pwd)"
if [[ "$CURRENT_DIR" == *"/build-"* ]]; then
    # Extract the build directory name (e.g., build-clangwarn)
    BUILD_DIR_NAME="$(basename "$CURRENT_DIR")"
    BUILD_DIR="$PROJECT_ROOT/$BUILD_DIR_NAME"
    print_info "Using build directory: $BUILD_DIR_NAME"
else
    BUILD_DIR="$PROJECT_ROOT/build"
fi

print_info "libstats Build Script"
print_info "Project Root: $PROJECT_ROOT"
print_info "Build Type: $BUILD_TYPE"

# Detect CPU count
if [ -n "$OVERRIDE_JOBS" ]; then
    CPU_COUNT="$OVERRIDE_JOBS"
    print_info "Using override: $CPU_COUNT parallel jobs"
else
    CPU_COUNT=$(detect_cpu_cores)
fi

# Clean build directory if requested
if [ "$CLEAN_BUILD" = true ]; then
    print_info "Cleaning build directory..."
    rm -rf "$BUILD_DIR"
fi

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure step
if [ "$BUILD_ONLY" != true ]; then
    print_info "Configuring build with CMake..."
    cmake_args=(
        -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
        "-DCMAKE_BUILD_PARALLEL_LEVEL=$CPU_COUNT"
    )

    if [ "$VERBOSE" = true ]; then
        cmake_args+=("--verbose")
    fi

    cmake "${cmake_args[@]}" "$PROJECT_ROOT"

    if [ $? -ne 0 ]; then
        print_error "CMake configuration failed"
        exit 1
    fi

    print_success "CMake configuration completed"
fi

# Build step
if [ "$CONFIGURE_ONLY" != true ]; then
    print_info "Building with $CPU_COUNT parallel jobs..."

    build_args=(
        --build .
        --parallel "$CPU_COUNT"
    )

    if [ "$VERBOSE" = true ]; then
        build_args+=("--verbose")
    fi

    cmake "${build_args[@]}"

    if [ $? -ne 0 ]; then
        print_error "Build failed"
        exit 1
    fi

    print_success "Build completed successfully"
fi

# Run tests if requested
if [ "$RUN_TESTS" = true ] && [ "$CONFIGURE_ONLY" != true ]; then
    print_info "Running tests..."
    ctest --output-on-failure --parallel "$CPU_COUNT"

    if [ $? -eq 0 ]; then
        print_success "All tests passed"
    else
        print_error "Some tests failed"
        exit 1
    fi
fi

print_success "Done!"

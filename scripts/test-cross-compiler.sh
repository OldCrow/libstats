#!/bin/bash

# Cross-Compiler Compatibility Testing Script for libstats
# Tests the code against multiple compiler strictness modes to catch compatibility issues

set -e  # Exit on any error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Results tracking
TOTAL_BUILDS=0
SUCCESSFUL_BUILDS=0
FAILED_BUILDS=0

print_header() {
    echo -e "${BLUE}============================================${NC}"
    echo -e "${BLUE}  libstats Cross-Compiler Compatibility Test${NC}"
    echo -e "${BLUE}============================================${NC}"
    echo
}

print_section() {
    echo -e "${YELLOW}>>> $1${NC}"
    echo
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# Clean function
cleanup_build_dirs() {
    print_section "Cleaning previous build directories"
    for config in MSVCStrict GCCStrict Release Debug; do
        build_dir="build-$(echo $config | tr '[:upper:]' '[:lower:]')"
        if [ -d "$build_dir" ]; then
            echo "  Removing $build_dir"
            rm -rf "$build_dir"
        fi
    done
    echo
}

# Test a specific build configuration
test_build_config() {
    local config="$1"
    local description="$2"
    local build_dir="build-$(echo $config | tr '[:upper:]' '[:lower:]')"
    
    print_section "Testing $config build"
    echo "Description: $description"
    echo "Build directory: $build_dir"
    echo
    
    TOTAL_BUILDS=$((TOTAL_BUILDS + 1))
    
    # Create build directory
    mkdir -p "$build_dir"
    cd "$build_dir"
    
    # Configure
    echo "Configuring..."
    if ! cmake -DCMAKE_BUILD_TYPE="$config" .. &>/dev/null; then
        print_error "CMake configuration failed for $config"
        cd ..
        return 1
    fi
    
    # Build (capture both stdout and stderr, show only first 50 lines of errors)
    echo "Building..."
    if make -j$(nproc 2>/dev/null || echo 2) >build.log 2>&1; then
        print_success "$config build completed successfully"
        SUCCESSFUL_BUILDS=$((SUCCESSFUL_BUILDS + 1))
        
        # Optional: Run a quick smoke test
        if find . -name "libstats*" -type f | grep -q .; then
            print_success "Generated library files found"
        else
            print_warning "No library files found (may be normal for object-only builds)"
        fi
    else
        print_error "$config build failed"
        FAILED_BUILDS=$((FAILED_BUILDS + 1))
        
        echo "Build errors (first 50 lines):"
        head -50 build.log | sed 's/^/  /'
        echo
        if [ "$(wc -l < build.log)" -gt 50 ]; then
            echo "  ... ($(wc -l < build.log) total lines in build.log)"
        fi
        
        # For strict modes, show which specific warnings caused failures
        if echo "$config" | grep -q "Strict"; then
            echo "Specific error analysis:"
            if grep -q "implicit conversion" build.log; then
                print_warning "  Found implicit conversion warnings"
            fi
            if grep -q "unused variable" build.log; then
                print_warning "  Found unused variable warnings"
            fi
            if grep -q "shadow" build.log; then
                print_warning "  Found variable shadowing warnings"
            fi
            if grep -q "old-style-cast" build.log; then
                print_warning "  Found old-style cast warnings"
            fi
            echo
        fi
    fi
    
    cd ..
    echo
}

# Print final summary
print_summary() {
    print_section "Cross-Compiler Compatibility Test Results"
    
    echo "Total builds tested: $TOTAL_BUILDS"
    echo "Successful builds: $SUCCESSFUL_BUILDS"
    echo "Failed builds: $FAILED_BUILDS"
    echo
    
    # Compatibility recommendations
    if [ "$FAILED_BUILDS" -eq 0 ]; then
        print_success "All builds passed! Your code is compatible across compiler configurations."
        echo "🎉 Ready for cross-platform deployment."
    else
        print_error "Some builds failed. Recommendations:"
        echo "  • Review error logs in build-*/build.log files"
        echo "  • Consider using explicit casts for type conversions"
        echo "  • Remove unused variables and parameters"
        echo "  • Use modern C++ casts instead of C-style casts"
    fi
    echo
}

# Show usage if requested
if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    echo "Cross-Compiler Compatibility Testing Script"
    echo
    echo "Usage: $0 [--clean] [--help]"
    echo
    echo "Options:"
    echo "  --clean    Remove all previous build directories before testing"
    echo "  --help     Show this help message"
    echo
    echo "This script tests the codebase against multiple compiler configurations:"
    echo "  • MSVCStrict: MSVC-like strictness (type conversions, implicit casts)"
    echo "  • GCCStrict: GCC-specific warnings (undefined behavior, duplicated conditions)"
    echo "  • Release: Standard release build (baseline compatibility)"
    echo "  • Debug: Standard debug build (baseline compatibility)"
    echo
    echo "The script helps catch compatibility issues before deploying to different"
    echo "platforms and compiler environments."
    exit 0
fi

# Main execution
main() {
    print_header
    
    # Clean previous builds if requested
    if [ "$1" == "--clean" ]; then
        cleanup_build_dirs
    fi
    
    # Test each configuration
    test_build_config "MSVCStrict" "MSVC-like strictness (type conversions, implicit casts) - ERRORS"
    test_build_config "MSVCWarn" "MSVC-like strictness (type conversions, implicit casts) - WARNINGS"
    test_build_config "GCCStrict" "GCC-specific warnings (undefined behavior, duplicated conditions) - ERRORS"
    test_build_config "GCCWarn" "GCC-specific warnings (undefined behavior, duplicated conditions) - WARNINGS"
    test_build_config "Release" "Standard release build (baseline compatibility)"
    test_build_config "Debug" "Standard debug build (baseline compatibility)"
    
    # Print summary
    print_summary
    
    # Exit with error code if any builds failed
    if [ "$FAILED_BUILDS" -gt 0 ]; then
        exit 1
    else
        exit 0
    fi
}

# Run main function
main "$@"

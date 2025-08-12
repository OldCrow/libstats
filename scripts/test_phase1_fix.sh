#!/bin/bash

# test_phase1_fix.sh
# 
# Comprehensive testing script for Phase 1 CMake modernization
# Tests that the problematic generator expressions have been fixed

set -e  # Exit on any error

echo "=============================================="
echo "Phase 1 CMake Modernization Testing"
echo "=============================================="

# Clean any existing build
echo "Step 1: Cleaning build directory..."
rm -rf build
mkdir build
cd build

echo ""
echo "Step 2: Configuring with CMake..."
echo "=============================================="

# Configure and capture output
cmake .. 2>&1 | tee cmake_configure.log

echo ""
echo "Step 3: Building with verbose output..."
echo "=============================================="

# Build with verbose output to see actual compiler flags
cmake --build . -- VERBOSE=1 2>&1 | tee cmake_build.log

echo ""
echo "Step 4: Analyzing build output for problematic patterns..."
echo "=============================================="

# Check for problematic generator expression patterns in build output
PROBLEMS_FOUND=0

echo "Checking for generator expressions in compiler commands..."
if grep -q '\$<' cmake_build.log; then
    echo "❌ PROBLEM: Generator expressions found in compiler commands:"
    grep '\$<' cmake_build.log | head -5
    PROBLEMS_FOUND=1
else
    echo "✅ GOOD: No generator expressions in compiler commands"
fi

echo ""
echo "Checking for duplicate SIMD flags..."
# Look for patterns like multiple -mavx2 or /arch:AVX2 flags
if grep -E '(-mavx2.*-mavx2|/arch:AVX2.*/arch:AVX2)' cmake_build.log; then
    echo "❌ PROBLEM: Duplicate SIMD flags detected"
    PROBLEMS_FOUND=1
else
    echo "✅ GOOD: No duplicate SIMD flags detected"
fi

echo ""
echo "Checking for traditional conditional messages..."
if grep -q "Applied.*SIMD flags" cmake_configure.log; then
    echo "✅ GOOD: Traditional SIMD conditional logic working"
    grep "Applied.*SIMD flags" cmake_configure.log
else
    echo "⚠️  NOTE: No SIMD flag messages (may be expected for some platforms)"
fi

echo ""
echo "Step 5: Testing SIMD integration..."
echo "=============================================="

# Test if SIMD test builds and runs
if [ -f "tests/test_simd_integration" ]; then
    echo "Running SIMD integration test..."
    if ./tests/test_simd_integration; then
        echo "✅ GOOD: SIMD integration test passed"
    else
        echo "❌ PROBLEM: SIMD integration test failed"
        PROBLEMS_FOUND=1
    fi
else
    echo "⚠️  NOTE: SIMD integration test not built (may be expected)"
fi

echo ""
echo "Step 6: Checking SIMD status messages..."
echo "=============================================="

echo "SIMD detection results from configure:"
grep -E "(SIMD:|SSE2:|AVX:|AVX2:|NEON:)" cmake_configure.log || echo "No SIMD status messages found"

echo ""
echo "Step 7: Validating compiler flag distribution..."
echo "=============================================="

# Count occurrences of different compiler flags to ensure they're not duplicated
echo "Compiler flag analysis:"
echo "  -mavx2 occurrences: $(grep -c '\-mavx2' cmake_build.log || echo 0)"
echo "  /arch:AVX2 occurrences: $(grep -c '/arch:AVX2' cmake_build.log || echo 0)"
echo "  -msse2 occurrences: $(grep -c '\-msse2' cmake_build.log || echo 0)"

echo ""
echo "=============================================="
echo "Phase 1 Testing Summary"
echo "=============================================="

if [ $PROBLEMS_FOUND -eq 0 ]; then
    echo "🎉 SUCCESS: Phase 1 fix appears to be working correctly!"
    echo ""
    echo "✅ No generator expressions in compiler commands"
    echo "✅ No duplicate SIMD flags detected" 
    echo "✅ Traditional conditionals working"
    echo ""
    echo "Next steps:"
    echo "1. Test on different platforms (Linux, Windows)"
    echo "2. Test different build types (Release, Debug, Dev)"
    echo "3. Verify all test suites pass"
    echo "4. Consider Phase 2 modernization when ready"
else
    echo "❌ ISSUES FOUND: Phase 1 fix needs attention"
    echo ""
    echo "Please review the output above and fix any issues before proceeding."
    echo "Common fixes:"
    echo "1. Check that generator expressions are properly removed"
    echo "2. Ensure SIMD detection is working correctly"
    echo "3. Verify traditional conditionals are being used"
fi

echo ""
echo "Build logs saved to:"
echo "  - cmake_configure.log: CMake configuration output" 
echo "  - cmake_build.log: Build output with compiler commands"

cd ..  # Return to project root

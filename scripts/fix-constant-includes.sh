#!/bin/bash

# Script to replace umbrella constants.h includes with specific constant headers
# Based on IWYU recommendations

echo "Fixing constant header includes based on IWYU analysis..."

# Fix simd_dispatch.cpp
echo "Fixing src/simd_dispatch.cpp..."
sed -i '' 's|#include "../include/core/constants.h"|// Removed umbrella constants.h - using specific headers|' src/simd_dispatch.cpp
# Add mathematical_constants.h after platform includes if not present
if ! grep -q "mathematical_constants.h" src/simd_dispatch.cpp; then
    sed -i '' '/#include.*platform_constants.h/a\
#include "../include/core/mathematical_constants.h"  // for ZERO_INT, FOUR_INT' src/simd_dispatch.cpp
fi

# Fix validation.cpp
echo "Fixing src/validation.cpp..."
sed -i '' 's|#include "../include/core/constants.h"|// Removed umbrella constants.h - using specific headers|' src/validation.cpp
# Add specific headers after distribution_base.h if not present
if ! grep -q "mathematical_constants.h" src/validation.cpp; then
    sed -i '' '/#include.*distribution_base.h/a\
#include "../include/core/mathematical_constants.h"  // for ZERO_DOUBLE, ONE, TWO\
#include "../include/core/threshold_constants.h"     // for ANDERSON_DARLING thresholds' src/validation.cpp
fi

# Fix gaussian.cpp
echo "Fixing src/gaussian.cpp..."
sed -i '' 's|#include "../include/core/constants.h"|// Removed umbrella constants.h - using specific headers|' src/gaussian.cpp

# Fix discrete.cpp
echo "Fixing src/discrete.cpp..."
sed -i '' 's|#include "../include/core/constants.h"|// Removed umbrella constants.h - using specific headers|' src/discrete.cpp

# Fix cpu_detection.cpp
echo "Fixing src/cpu_detection.cpp..."
sed -i '' 's|#include "../include/core/constants.h"|// Removed umbrella constants.h - using specific headers|' src/cpu_detection.cpp
# Add mathematical_constants.h if not present
if ! grep -q "mathematical_constants.h" src/cpu_detection.cpp; then
    sed -i '' '/#include.*platform_constants.h/a\
#include "../include/core/mathematical_constants.h"  // for ZERO_INT, ONE_INT, TWO_INT' src/cpu_detection.cpp
fi

# Fix poisson.cpp
echo "Fixing src/poisson.cpp..."
sed -i '' 's|#include "../include/core/constants.h"|// Removed umbrella constants.h - using specific headers|' src/poisson.cpp

# Fix safety.cpp
echo "Fixing src/safety.cpp..."
sed -i '' 's|#include "../include/core/constants.h"|// Removed umbrella constants.h - using specific headers|' src/safety.cpp
# Add mathematical_constants.h if not present
if ! grep -q "mathematical_constants.h" src/safety.cpp; then
    sed -i '' '/#include.*core\/safety.h/a\
#include "../include/core/mathematical_constants.h"  // for ZERO_DOUBLE, ONE' src/safety.cpp
fi

# Fix simd_avx.cpp
echo "Fixing src/simd_avx.cpp..."
sed -i '' 's|#include "../include/core/constants.h"|// Removed umbrella constants.h - using specific headers|' src/simd_avx.cpp

# Fix simd_fallback.cpp
echo "Fixing src/simd_fallback.cpp..."
sed -i '' 's|#include "../include/core/constants.h"|// Removed umbrella constants.h - using specific headers|' src/simd_fallback.cpp

# Fix distribution_base.cpp
echo "Fixing src/distribution_base.cpp..."
sed -i '' 's|#include "../include/core/constants.h"|// Removed umbrella constants.h - using specific headers|' src/distribution_base.cpp

# Fix exponential.cpp
echo "Fixing src/exponential.cpp..."
sed -i '' 's|#include "../include/core/constants.h"|// Removed umbrella constants.h - using specific headers|' src/exponential.cpp

# Fix simd_sse2.cpp
echo "Fixing src/simd_sse2.cpp..."
sed -i '' 's|#include "../include/core/constants.h"|// Removed umbrella constants.h - using specific headers|' src/simd_sse2.cpp

# Fix math_utils.cpp
echo "Fixing src/math_utils.cpp..."
sed -i '' 's|#include "../include/core/constants.h"|// Removed umbrella constants.h - using specific headers|' src/math_utils.cpp

# Fix other source files that might have the issue
for file in src/*.cpp; do
    if grep -q '#include "../include/core/constants.h"' "$file"; then
        echo "Fixing $file..."
        sed -i '' 's|#include "../include/core/constants.h"|// Removed umbrella constants.h - using specific headers|' "$file"
    fi
done

echo "Done! Now running build to check for compilation errors..."
cmake --build build --parallel 8

echo ""
echo "If build succeeds, run IWYU again to verify improvements:"
echo "./scripts/run-iwyu.sh --src 2>/dev/null | grep 'constants.h' | wc -l"

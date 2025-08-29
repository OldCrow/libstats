#!/bin/bash

# Script to separate architecture-specific constants into stats::arch::
# while keeping general constants in stats::detail::

echo "=== Separating architecture-specific constants into stats::arch:: ==="
echo "Architecture-specific: SIMD, CPU, parallel, adaptive execution"
echo "General details: mathematical, statistical, probability constants"
echo ""

# First, let's identify and update platform/architecture-specific constant files
echo "=== Step 1: Update platform_constants.h to use stats::arch:: ==="

if [ -f "include/platform/platform_constants.h" ]; then
    echo "Processing include/platform/platform_constants.h..."

    # Update platform_constants.h to use arch namespace instead of detail
    perl -i -pe '
        # Change namespace detail to namespace arch for platform constants
        s/^namespace detail\s*\{/namespace arch {/;
        s/^\}\s*\/\/\s*namespace detail/}  \/\/ namespace arch/;
    ' "include/platform/platform_constants.h"

    echo "  Updated platform_constants.h to use stats::arch::"
fi

echo ""
echo "=== Step 2: Update all references in the codebase ==="

# Find all C++ source and header files and update references
find . -type f \( -name "*.cpp" -o -name "*.h" -o -name "*.hpp" \) \
    -not -path "./build/*" \
    -not -path "./.git/*" \
    -not -path "./cmake-build-*/*" \
    -not -name "*.bak" | while read -r file; do

    # Update references to properly route to arch:: or detail::
    perl -i -pe '
        # First pass: Fix any remaining constants:: references
        # These should all go to either detail:: or arch:: based on context

        # Architecture-specific patterns -> arch::
        # SIMD and CPU related
        s/\bconstants::simd::cpu::/arch::/g;
        s/\bconstants::simd::/arch::/g;
        s/\bconstants::cpu::/arch::/g;
        s/\bstats::constants::simd::cpu::/stats::arch::/g;
        s/\bstats::constants::simd::/stats::arch::/g;
        s/\bstats::constants::cpu::/stats::arch::/g;

        # Parallel execution related
        s/\bconstants::parallel::adaptive::/arch::/g;
        s/\bconstants::parallel::/arch::/g;
        s/\bconstants::adaptive::/arch::/g;
        s/\bstats::constants::parallel::adaptive::/stats::arch::/g;
        s/\bstats::constants::parallel::/stats::arch::/g;
        s/\bstats::constants::adaptive::/stats::arch::/g;

        # Platform-specific constants
        s/\bconstants::platform::/arch::/g;
        s/\bstats::constants::platform::/stats::arch::/g;

        # Architecture/hardware related
        s/\bconstants::hardware::/arch::/g;
        s/\bstats::constants::hardware::/stats::arch::/g;
        s/\bconstants::cache::/arch::/g;
        s/\bstats::constants::cache::/stats::arch::/g;

        # Fix detail:: references that should be arch::
        s/\bdetail::simd::cpu::/arch::/g;
        s/\bdetail::simd::/arch::/g;
        s/\bdetail::cpu::/arch::/g;
        s/\bdetail::parallel::adaptive::/arch::/g;
        s/\bdetail::parallel::/arch::/g;
        s/\bdetail::adaptive::/arch::/g;
        s/\bdetail::platform::/arch::/g;
        s/\bdetail::hardware::/arch::/g;
        s/\bdetail::cache::/arch::/g;

        s/\bstats::detail::simd::cpu::/stats::arch::/g;
        s/\bstats::detail::simd::/stats::arch::/g;
        s/\bstats::detail::cpu::/stats::arch::/g;
        s/\bstats::detail::parallel::adaptive::/stats::arch::/g;
        s/\bstats::detail::parallel::/stats::arch::/g;
        s/\bstats::detail::adaptive::/stats::arch::/g;
        s/\bstats::detail::platform::/stats::arch::/g;
        s/\bstats::detail::hardware::/stats::arch::/g;
        s/\bstats::detail::cache::/stats::arch::/g;

        # General catch-all for remaining constants:: -> detail::
        s/\bstats::constants::/stats::detail::/g;
        s/\bconstants::/detail::/g;

        # Fix using namespace declarations
        s/using namespace stats::constants;/using namespace stats::detail;/g;
        s/using namespace constants;/using namespace detail;/g;
    ' "$file"
done

echo ""
echo "=== Step 3: Check for specific architecture constant names and update them ==="

# Now look for specific constant names that should be in arch::
ARCH_CONSTANT_PATTERNS=(
    "NANOSECONDS_TO_HZ"
    "grain_size"
    "min_elements_for_parallel"
    "min_elements_for_distribution_parallel"
    "simple_operation_grain_size"
    "complex_operation_grain_size"
    "CACHE_LINE_SIZE"
    "L1_CACHE_SIZE"
    "L2_CACHE_SIZE"
    "PREFETCH_DISTANCE"
    "VECTOR_WIDTH"
    "MAX_THREADS"
    "MIN_PARALLEL_SIZE"
)

for pattern in "${ARCH_CONSTANT_PATTERNS[@]}"; do
    echo "Updating references to $pattern..."
    find . -type f \( -name "*.cpp" -o -name "*.h" -o -name "*.hpp" \) \
        -not -path "./build/*" \
        -not -path "./.git/*" \
        -not -path "./cmake-build-*/*" \
        -not -name "*.bak" | xargs perl -i -pe "
        # Update detail::pattern to arch::pattern
        s/\\bdetail::${pattern}/arch::${pattern}/g;
        s/\\bstats::detail::${pattern}/stats::arch::${pattern}/g;
    "
done

echo ""
echo "=== Step 4: Look for files that might need arch namespace declarations ==="

# Find files that likely contain architecture-specific code
echo "Files that might need stats::arch namespace:"
grep -l "SIMD\|parallel\|cpu_features\|cache\|thread" \
    include/platform/*.h include/core/*parallel*.h 2>/dev/null | head -10

echo ""
echo "=== Separation complete! ==="
echo ""
echo "Summary:"
echo "1. Architecture-specific constants moved to stats::arch::"
echo "   - SIMD, CPU detection, cache sizes"
echo "   - Parallel execution parameters"
echo "   - Platform-specific optimizations"
echo ""
echo "2. General constants remain in stats::detail::"
echo "   - Mathematical constants"
echo "   - Statistical thresholds"
echo "   - Probability values"
echo ""
echo "Next step: Compile to verify the separation is correct"

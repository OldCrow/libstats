#!/bin/bash

# Script to flatten the stats::arch namespace hierarchy
# Keep meaningful prefixes where necessary to avoid naming collisions

echo "=== Flattening stats::arch namespace hierarchy ==="
echo ""

# First, update platform_constants.h to flatten nested namespaces
echo "Step 1: Flattening platform_constants.h namespaces..."

if [ -f "include/platform/platform_constants.h" ]; then
    # Create a backup
    cp "include/platform/platform_constants.h" "include/platform/platform_constants.h.bak2"

    # Flatten the nested namespaces but keep prefixes for clarity
    perl -i -pe '
        # Track namespace depth
        BEGIN { $in_arch = 0; }

        # Mark when we enter arch namespace
        $in_arch = 1 if /^namespace arch\s*\{/;
        $in_arch = 0 if /^\}\s*\/\/\s*namespace arch/;

        if ($in_arch) {
            # Comment out nested namespace declarations within arch
            s/^namespace (simd|parallel|memory|sse|avx|avx2|avx512|neon|fallback|adaptive|tuning|prefetch|distance|alignment|matrix|registers|unroll|cpu|optimization|batch_sizes)\s*\{/\/\/ Flattened: was namespace $1/g;

            # Comment out nested namespace closings
            s/^\}\s*\/\/\s*namespace (simd|parallel|memory|sse|avx|avx2|avx512|neon|fallback|adaptive|tuning|prefetch|distance|alignment|matrix|registers|unroll|cpu|optimization|batch_sizes)/\/\/ End flattened $1/g;

            # Special handling for nested functions in adaptive namespace - they stay as functions
            # No changes needed for function definitions
        }
    ' "include/platform/platform_constants.h"

    echo "  Done flattening platform_constants.h"
fi

echo ""
echo "Step 2: Updating all references to use flattened arch namespace..."

# Now update all references throughout the codebase
find . -type f \( -name "*.cpp" -o -name "*.h" -o -name "*.hpp" \) \
    -not -path "./build/*" \
    -not -path "./.git/*" \
    -not -path "./cmake-build-*/*" \
    -not -name "*.bak*" | while read -r file; do

    perl -i -pe '
        # Update nested arch references to flat arch:: with appropriate prefixes

        # SIMD CPU constants - add prefix to avoid collision
        s/\barch::simd::cpu::NANOSECONDS_TO_HZ\b/arch::NANOSECONDS_TO_HZ/g;
        s/\barch::simd::cpu::([A-Z_]+)\b/arch::$1/g;

        # Parallel adaptive functions - these are functions, not constants
        s/\barch::parallel::adaptive::grain_size\(\)/arch::adaptive::grain_size()/g;
        s/\barch::parallel::adaptive::min_elements_for_parallel\(\)/arch::adaptive::min_elements_for_parallel()/g;
        s/\barch::parallel::adaptive::min_elements_for_distribution_parallel\(\)/arch::adaptive::min_elements_for_distribution_parallel()/g;
        s/\barch::parallel::adaptive::min_elements_for_simple_distribution_parallel\(\)/arch::adaptive::min_elements_for_simple_distribution_parallel()/g;
        s/\barch::parallel::adaptive::simple_operation_grain_size\(\)/arch::adaptive::simple_operation_grain_size()/g;
        s/\barch::parallel::adaptive::complex_operation_grain_size\(\)/arch::adaptive::complex_operation_grain_size()/g;
        s/\barch::parallel::adaptive::monte_carlo_grain_size\(\)/arch::adaptive::monte_carlo_grain_size()/g;
        s/\barch::parallel::adaptive::max_grain_size\(\)/arch::adaptive::max_grain_size()/g;

        # Now update without the parallel:: prefix
        s/\barch::grain_size\(\)/arch::adaptive::grain_size()/g;
        s/\barch::min_elements_for_parallel\(\)/arch::adaptive::min_elements_for_parallel()/g;
        s/\barch::min_elements_for_distribution_parallel\(\)/arch::adaptive::min_elements_for_distribution_parallel()/g;
        s/\barch::simple_operation_grain_size\(\)/arch::adaptive::simple_operation_grain_size()/g;
        s/\barch::complex_operation_grain_size\(\)/arch::adaptive::complex_operation_grain_size()/g;

        # Architecture-specific constants (sse::, avx::, etc) - add architecture prefix
        s/\barch::parallel::sse::([A-Z_]+)\b/arch::SSE_$1/g;
        s/\barch::parallel::avx::([A-Z_]+)\b/arch::AVX_$1/g;
        s/\barch::parallel::avx2::([A-Z_]+)\b/arch::AVX2_$1/g;
        s/\barch::parallel::avx512::([A-Z_]+)\b/arch::AVX512_$1/g;
        s/\barch::parallel::neon::([A-Z_]+)\b/arch::NEON_$1/g;
        s/\barch::parallel::fallback::([A-Z_]+)\b/arch::FALLBACK_$1/g;

        # SIMD constants
        s/\barch::simd::([A-Z_]+)\b/arch::SIMD_$1/g;
        s/\barch::simd::alignment::([A-Z_]+)\b/arch::$1/g;
        s/\barch::simd::matrix::([A-Z_]+)\b/arch::MATRIX_$1/g;
        s/\barch::simd::registers::([A-Z_]+)\b/arch::$1/g;
        s/\barch::simd::unroll::([A-Z_]+)\b/arch::$1/g;
        s/\barch::simd::optimization::([A-Z_]+)\b/arch::$1/g;

        # Parallel batch sizes
        s/\barch::parallel::batch_sizes::([A-Z_]+)\b/arch::BATCH_$1/g;

        # Memory prefetch constants
        s/\barch::memory::prefetch::distance::([A-Z_]+)\b/arch::PREFETCH_$1/g;

        # Stats arch function calls
        s/\bstats::arch::grain_size\(\)/stats::arch::adaptive::grain_size()/g;
        s/\bstats::arch::min_elements_for_parallel\(\)/stats::arch::adaptive::min_elements_for_parallel()/g;
        s/\bstats::arch::min_elements_for_distribution_parallel\(\)/stats::arch::adaptive::min_elements_for_distribution_parallel()/g;
        s/\bstats::arch::simple_operation_grain_size\(\)/stats::arch::adaptive::simple_operation_grain_size()/g;
        s/\bstats::arch::complex_operation_grain_size\(\)/stats::arch::adaptive::complex_operation_grain_size()/g;
    ' "$file"
done

echo ""
echo "Step 3: Checking for any remaining nested namespace references..."

# Look for any remaining nested references
echo "Checking for remaining nested arch:: patterns..."
grep -r "arch::.*::.*::" . \
    --include="*.cpp" \
    --include="*.h" \
    --include="*.hpp" \
    --exclude-dir=build \
    --exclude-dir=.git \
    --exclude="*.bak*" | head -10 || echo "No remaining deeply nested arch:: references found!"

echo ""
echo "=== Flattening complete! ==="
echo ""
echo "Summary:"
echo "1. Flattened nested namespaces within stats::arch::"
echo "2. Added prefixes where necessary to avoid naming collisions"
echo "3. Preserved adaptive:: sub-namespace for runtime functions"
echo ""
echo "Next step: Compile to verify the changes"

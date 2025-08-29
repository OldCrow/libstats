#!/bin/bash

# Complete flattening of arch namespace - no sub-namespaces at all
echo "=== Complete flattening of arch namespace ==="
echo ""

# First restore from backup to start fresh
echo "Step 1: Restoring platform_constants.h from backup..."
if [ -f "include/platform/platform_constants.h.bak2" ]; then
    cp "include/platform/platform_constants.h.bak2" "include/platform/platform_constants.h"
    echo "  Restored"
fi

# Now flatten EVERYTHING in arch namespace
echo "Step 2: Flattening all nested namespaces in arch..."

perl -i -pe '
    BEGIN {
        $in_arch = 0;
        $section = "";
    }

    # Track arch namespace
    $in_arch = 1 if /^namespace arch\s*\{/;
    $in_arch = 0 if /^\}\s*\/\/\s*namespace arch/;

    if ($in_arch) {
        # Track sections for prefixing
        if (/namespace (\w+)\s*\{/) {
            my $ns = $1;
            # Determine the prefix based on namespace
            if ($ns eq "sse" || $ns eq "avx" || $ns eq "avx2" || $ns eq "avx512" || $ns eq "neon" || $ns eq "fallback") {
                $section = uc($ns) . "_";
            } elsif ($ns eq "alignment") {
                $section = "";  # These are unique enough
            } elsif ($ns eq "matrix") {
                $section = "MATRIX_";
            } elsif ($ns eq "registers") {
                $section = "";  # Already have good names like AVX_DOUBLES
            } elsif ($ns eq "unroll") {
                $section = "";  # Already have good names like AVX_UNROLL
            } elsif ($ns eq "cpu") {
                $section = "";  # These are unique
            } elsif ($ns eq "optimization") {
                $section = "";  # These are unique
            } elsif ($ns eq "batch_sizes") {
                $section = "";  # Already prefixed with SMALL_BATCH etc
            } elsif ($ns eq "adaptive") {
                $section = ""; # These are functions, not constants
            } elsif ($ns eq "tuning") {
                $section = "";  # These are unique
            } elsif ($ns eq "prefetch" || $ns eq "distance") {
                $section = "PREFETCH_";
            } else {
                $section = uc($ns) . "_";
            }

            # Comment out the namespace declaration
            $_ = "// Flattened: was namespace $ns\n";
        }

        # Remove namespace closing braces
        if (/^\}\s*\/\/\s*namespace (\w+)/) {
            $_ = "// End $1 section\n";
            # Clear section unless we are in a nested architecture-specific section
            if ($1 eq "legacy_intel") {
                # Keep section for legacy_intel within avx
            } else {
                $section = "" unless $1 eq "distributions";
            }
        }

        # Add prefixes to constants
        if ($section && /^inline constexpr .* ([A-Z_]+)\s*=/) {
            my $const = $1;
            # Only add prefix if not already present
            unless ($const =~ /^(SSE|AVX|AVX2|AVX512|NEON|FALLBACK|MATRIX|PREFETCH)_/) {
                s/^(inline constexpr .* )([A-Z_]+)(\s*=)/${1}${section}${2}${3}/;
            }
        }

        # Handle adaptive functions - these should be directly in arch::
        if (/^inline std::size_t (\w+)\(/) {
            # Function declarations stay as-is
        }

        # Fix function bodies that reference nested namespaces
        s/return (\w+)::([A-Z_]+);/return ${1}_${2};/g if /return.*::/;
    }
' include/platform/platform_constants.h

echo "  Flattened"

# Step 3: Update all references throughout the codebase
echo "Step 3: Updating all references..."

find . -type f \( -name "*.cpp" -o -name "*.h" -o -name "*.hpp" \) \
    -not -path "./build/*" \
    -not -path "./.git/*" \
    -not -path "./cmake-build-*/*" \
    -not -name "*.bak*" | while read -r file; do

    perl -i -pe '
        # Remove all nested namespace references in arch

        # Functions that were in adaptive:: now directly in arch::
        s/\barch::adaptive::/arch::/g;
        s/\bstats::arch::adaptive::/stats::arch::/g;

        # Constants from nested namespaces
        s/\barch::registers::([A-Z_]+)/arch::$1/g;
        s/\barch::unroll::([A-Z_]+)/arch::$1/g;
        s/\barch::optimization::([A-Z_]+)/arch::$1/g;
        s/\barch::cpu::([A-Z_]+)/arch::$1/g;
        s/\barch::alignment::([A-Z_]+)/arch::$1/g;
        s/\barch::matrix::([A-Z_]+)/arch::MATRIX_$1/g;
        s/\barch::batch_sizes::([A-Z_]+)/arch::$1/g;
        s/\barch::tuning::([A-Z_]+)/arch::$1/g;
        s/\barch::simd::([A-Z_]+)/arch::$1/g;
        s/\barch::prefetch::distance::([A-Z_]+)/arch::PREFETCH_$1/g;
        s/\barch::memory::prefetch::distance::([A-Z_]+)/arch::PREFETCH_$1/g;

        # Architecture-specific constants
        s/\barch::sse::([A-Z_]+)/arch::SSE_$1/g;
        s/\barch::avx::([A-Z_]+)/arch::AVX_$1/g;
        s/\barch::avx2::([A-Z_]+)/arch::AVX2_$1/g;
        s/\barch::avx512::([A-Z_]+)/arch::AVX512_$1/g;
        s/\barch::neon::([A-Z_]+)/arch::NEON_$1/g;
        s/\barch::fallback::([A-Z_]+)/arch::FALLBACK_$1/g;

        # Special case for parallel namespace
        s/\barch::parallel::([A-Z_]+)/arch::$1/g;
        s/\bstats::arch::parallel::/stats::arch::/g;

        # Move functions from detail:: to arch:: where appropriate
        s/\bdetail::get_optimal_simd_block_size\(/arch::get_optimal_simd_block_size(/g;
        s/\bdetail::get_min_simd_size\(/arch::get_min_simd_size(/g;
        s/\bdetail::get_optimal_alignment\(/arch::get_optimal_alignment(/g;
        s/\bdetail::get_cache_thresholds\(/arch::get_cache_thresholds(/g;

        # Fix stats::detail:: references
        s/\bstats::detail::get_optimal_simd_block_size\(/stats::arch::get_optimal_simd_block_size(/g;
        s/\bstats::detail::get_min_simd_size\(/stats::arch::get_min_simd_size(/g;
        s/\bstats::detail::get_optimal_alignment\(/stats::arch::get_optimal_alignment(/g;
        s/\bstats::detail::get_cache_thresholds\(/stats::arch::get_cache_thresholds(/g;

        # Fix platform:: references
        s/\bplatform::get_optimal_simd_block_size\(/arch::get_optimal_simd_block_size(/g;

        # Fix using namespace declarations
        s/using namespace stats::parallel;/\/\/ Removed: using namespace stats::parallel;/g;
    ' "$file"
done

echo "  Updated"

# Step 4: Fix the adaptive functions in platform_constants.h
echo "Step 4: Fixing adaptive function implementations..."

perl -i -pe '
    # Fix return statements in adaptive functions
    s/return AVX512_([A-Z_]+);/return AVX512_$1;/g;
    s/return AVX2_([A-Z_]+);/return AVX2_$1;/g;
    s/return AVX_([A-Z_]+);/return AVX_$1;/g;
    s/return SSE_([A-Z_]+);/return SSE_$1;/g;
    s/return NEON_([A-Z_]+);/return NEON_$1;/g;
    s/return FALLBACK_([A-Z_]+);/return FALLBACK_$1;/g;

    # Fix legacy intel references
    s/return avx::legacy_intel::([A-Z_]+);/return AVX_$1; \/\/ Using AVX fallback for legacy Intel/g;
' include/platform/platform_constants.h

echo "  Fixed"

echo ""
echo "=== Complete flattening done! ==="
echo ""
echo "Summary:"
echo "1. ALL sub-namespaces removed from arch::"
echo "2. Architecture-specific constants prefixed (SSE_, AVX_, etc.)"
echo "3. Adaptive functions moved directly to arch::"
echo "4. Other constants kept with descriptive names"
echo "5. Updated all references throughout codebase"
echo ""
echo "Next: Compile to verify"

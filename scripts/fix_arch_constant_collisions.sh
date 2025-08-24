#!/bin/bash

# Script to fix naming collisions in arch namespace by adding proper prefixes

echo "=== Fixing architecture constant naming collisions ==="
echo ""

# Update platform_constants.h to add prefixes to architecture-specific constants
echo "Adding prefixes to architecture-specific constants..."

if [ -f "include/platform/platform_constants.h" ]; then
    perl -i -pe '
        # Track which section we are in
        BEGIN {
            $section = "";
            $in_arch = 0;
        }

        # Mark when we enter/exit arch namespace
        $in_arch = 1 if /^namespace arch\s*\{/;
        $in_arch = 0 if /^\}\s*\/\/\s*namespace arch/;

        if ($in_arch) {
            # Track which architecture section we are in
            if (/\/\/ Flattened: was namespace (\w+)/) {
                $section = uc($1);
                # Special cases for nested sections
                $section = "" if $section eq "SIMD" || $section eq "PARALLEL" || $section eq "MEMORY";
                $section = "" if $section eq "ADAPTIVE";  # Adaptive contains functions, not constants
            }

            # For SSE section
            if (/\/\/ ===== SSE\/SSE2 Architecture/ || /namespace sse/) {
                $section = "SSE";
            }

            # For AVX section
            if (/\/\/ ===== AVX Architecture/ && !/AVX2/ && !/AVX512/) {
                $section = "AVX";
            }

            # For AVX2 section
            if (/\/\/ ===== AVX2 Architecture/ || /namespace avx2/) {
                $section = "AVX2";
            }

            # For AVX512 section
            if (/\/\/ ===== AVX-512 Architecture/ || /namespace avx512/) {
                $section = "AVX512";
            }

            # For NEON section
            if (/\/\/ ===== ARM NEON Architecture/ || /namespace neon/) {
                $section = "NEON";
            }

            # For Fallback section
            if (/\/\/ ===== Fallback Constants/ || /namespace fallback/) {
                $section = "FALLBACK";
            }

            # Clear section on certain markers
            if (/\/\/ ===== Legacy Constants/ || /\/\/ ===== Platform-optimized functions/) {
                $section = "";
            }

            # Add prefixes to constant definitions based on section
            if ($section && /^inline constexpr std::size_t ([A-Z_]+)\s*=/) {
                my $const_name = $1;
                # Skip if already prefixed
                unless ($const_name =~ /^(SSE|AVX|AVX2|AVX512|NEON|FALLBACK)_/) {
                    s/^inline constexpr std::size_t ([A-Z_]+)/inline constexpr std::size_t ${section}_$1/;
                }
            }
        }
    ' "include/platform/platform_constants.h"

    echo "  Prefixes added to platform_constants.h"
fi

echo ""
echo "Now updating adaptive function implementations to use prefixed constants..."

# Update the adaptive functions to use the new prefixed constants
perl -i -pe '
    # In adaptive functions, update references to use prefixed constants

    # min_elements_for_parallel function
    s/return avx512::MIN_ELEMENTS_FOR_PARALLEL;/return AVX512_MIN_ELEMENTS_FOR_PARALLEL;/g;
    s/return avx2::MIN_ELEMENTS_FOR_PARALLEL;/return AVX2_MIN_ELEMENTS_FOR_PARALLEL;/g;
    s/return avx::legacy_intel::MIN_ELEMENTS_FOR_PARALLEL;/return AVX_LEGACY_INTEL_MIN_ELEMENTS_FOR_PARALLEL;/g;
    s/return avx::MIN_ELEMENTS_FOR_PARALLEL;/return AVX_MIN_ELEMENTS_FOR_PARALLEL;/g;
    s/return sse::MIN_ELEMENTS_FOR_PARALLEL;/return SSE_MIN_ELEMENTS_FOR_PARALLEL;/g;
    s/return neon::MIN_ELEMENTS_FOR_PARALLEL;/return NEON_MIN_ELEMENTS_FOR_PARALLEL;/g;
    s/return fallback::MIN_ELEMENTS_FOR_PARALLEL;/return FALLBACK_MIN_ELEMENTS_FOR_PARALLEL;/g;

    # min_elements_for_distribution_parallel function
    s/return avx512::MIN_ELEMENTS_FOR_DISTRIBUTION_PARALLEL;/return AVX512_MIN_ELEMENTS_FOR_DISTRIBUTION_PARALLEL;/g;
    s/return avx2::MIN_ELEMENTS_FOR_DISTRIBUTION_PARALLEL;/return AVX2_MIN_ELEMENTS_FOR_DISTRIBUTION_PARALLEL;/g;
    s/return avx::legacy_intel::MIN_ELEMENTS_FOR_DISTRIBUTION_PARALLEL;/return AVX_LEGACY_INTEL_MIN_ELEMENTS_FOR_DISTRIBUTION_PARALLEL;/g;
    s/return avx::MIN_ELEMENTS_FOR_DISTRIBUTION_PARALLEL;/return AVX_MIN_ELEMENTS_FOR_DISTRIBUTION_PARALLEL;/g;
    s/return sse::MIN_ELEMENTS_FOR_DISTRIBUTION_PARALLEL;/return SSE_MIN_ELEMENTS_FOR_DISTRIBUTION_PARALLEL;/g;
    s/return neon::MIN_ELEMENTS_FOR_DISTRIBUTION_PARALLEL;/return NEON_MIN_ELEMENTS_FOR_DISTRIBUTION_PARALLEL;/g;
    s/return fallback::MIN_ELEMENTS_FOR_DISTRIBUTION_PARALLEL;/return FALLBACK_MIN_ELEMENTS_FOR_DISTRIBUTION_PARALLEL;/g;

    # Similar for other adaptive functions
    s/return (\w+)::DEFAULT_GRAIN_SIZE;/return ${1}_DEFAULT_GRAIN_SIZE;/g;
    s/return (\w+)::SIMPLE_OPERATION_GRAIN_SIZE;/return ${1}_SIMPLE_OPERATION_GRAIN_SIZE;/g;
    s/return (\w+)::COMPLEX_OPERATION_GRAIN_SIZE;/return ${1}_COMPLEX_OPERATION_GRAIN_SIZE;/g;
    s/return (\w+)::MONTE_CARLO_GRAIN_SIZE;/return ${1}_MONTE_CARLO_GRAIN_SIZE;/g;
    s/return (\w+)::MAX_GRAIN_SIZE;/return ${1}_MAX_GRAIN_SIZE;/g;
    s/return (\w+)::MIN_ELEMENTS_FOR_SIMPLE_DISTRIBUTION_PARALLEL;/return ${1}_MIN_ELEMENTS_FOR_SIMPLE_DISTRIBUTION_PARALLEL;/g;

    # Fix avx::legacy_intel special case
    s/avx::legacy_intel::/AVX_LEGACY_INTEL_/g;
' include/platform/platform_constants.h

echo ""
echo "=== Fixing complete! ==="
echo ""
echo "Summary:"
echo "1. Added architecture prefixes (SSE_, AVX_, AVX2_, AVX512_, NEON_, FALLBACK_)"
echo "2. Updated adaptive functions to use prefixed constants"
echo "3. Avoided naming collisions in flattened arch namespace"
echo ""
echo "Next step: Compile to verify the fixes"

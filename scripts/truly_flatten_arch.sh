#!/bin/bash

# Script to TRULY flatten the arch namespace - no sub-namespaces at all
# Everything in arch:: gets prefixed appropriately to avoid collisions

echo "=== Truly flattening arch namespace - NO sub-namespaces ==="
echo ""

# Step 1: Fix platform_constants.h to have everything directly in arch::
echo "Step 1: Flattening ALL nested namespaces in platform_constants.h..."

perl -i -pe '
    BEGIN {
        $in_arch = 0;
        $section = "";
    }

    # Track arch namespace
    $in_arch = 1 if /^namespace arch\s*\{/;
    $in_arch = 0 if /^\}\s*\/\/\s*namespace arch/;

    if ($in_arch) {
        # Remove ALL nested namespace declarations
        if (/^namespace (\w+)\s*\{/) {
            $section = uc($1);
            $_ = "// Flattened into arch:: - was namespace $1\n";
        }

        # Remove namespace closing braces
        if (/^\}\s*\/\/\s*namespace (\w+)/) {
            $_ = "// End of $1 section\n";
            $section = "";
        }

        # Add prefixes to constants based on section
        if ($section && /^inline constexpr/) {
            # Special handling for different sections
            if ($section eq "ALIGNMENT") {
                # Keep these without prefix as they are unique enough
            } elsif ($section eq "MATRIX") {
                s/inline constexpr st

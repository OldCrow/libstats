#!/bin/bash

# Script to consolidate ALL constants files into stats::detail:: namespace
# This is part of Phase 2 aggressive namespace consolidation

echo "=== Consolidating ALL constants into stats::detail:: namespace ==="

# List of all constants files to update
CONSTANTS_FILES=(
    "include/core/benchmark_constants.h"
    "include/core/essential_constants.h"
    "include/core/goodness_of_fit_constants.h"
    "include/core/mathematical_constants.h"
    "include/core/precision_constants.h"
    "include/core/probability_constants.h"
    "include/core/robust_constants.h"
    "include/core/statistical_constants.h"
    "include/core/statistical_methods_constants.h"
    "include/core/threshold_constants.h"
    "include/platform/platform_constants.h"
)

# Process each constants file
for file in "${CONSTANTS_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "Processing $file..."

        # Create backup
        cp "$file" "${file}.bak"

        # Remove all nested namespace declarations within stats namespace
        # This handles various patterns like constants::math::, constants::probability::, etc.
        perl -i -pe '
            # Track if we are inside stats namespace
            $in_stats = 1 if /^namespace stats\s*\{/;
            $in_stats = 0 if /^\}\s*\/\/\s*namespace stats/;

            # If inside stats namespace, replace nested namespace declarations with detail
            if ($in_stats) {
                # Replace opening of nested namespaces with detail
                s/^namespace constants\s*\{\s*$/namespace detail {/;

                # Remove or comment out nested namespace declarations
                s/^namespace (math|probability|precision|benchmark|thresholds|chi_squared|goodness_of_fit|robust|statistical|platform|poisson)\s*\{/\/\/ Consolidated into detail namespace (was: namespace $1)/;

                # Remove corresponding closing braces with namespace comments
                s/^\}\s*\/\/\s*namespace (math|probability|precision|benchmark|thresholds|chi_squared|goodness_of_fit|robust|statistical|platform|poisson)/\/\/ End of consolidated $1 constants/;

                # Update the constants namespace closing to detail
                s/^\}\s*\/\/\s*namespace constants/\}  \/\/ namespace detail/;
            }
        ' "$file"

        echo "  Done: $file"
    else
        echo "  Warning: $file not found"
    fi
done

echo ""
echo "=== Now updating all references to the old namespaces ==="

# Find all C++ source and header files
find . -type f \( -name "*.cpp" -o -name "*.h" -o -name "*.hpp" \) \
    -not -path "./build/*" \
    -not -path "./.git/*" \
    -not -path "./cmake-build-*/*" | while read -r file; do

    # Skip backup files
    if [[ "$file" == *.bak ]]; then
        continue
    fi

    # Update references from old nested namespaces to stats::detail::
    perl -i -pe '
        # Update various nested namespace references to detail::
        s/\bstats::constants::math::/stats::detail::/g;
        s/\bstats::constants::probability::/stats::detail::/g;
        s/\bstats::constants::precision::/stats::detail::/g;
        s/\bstats::constants::benchmark::/stats::detail::/g;
        s/\bstats::constants::thresholds::/stats::detail::/g;
        s/\bstats::constants::chi_squared::/stats::detail::/g;
        s/\bstats::constants::goodness_of_fit::/stats::detail::/g;
        s/\bstats::constants::robust::/stats::detail::/g;
        s/\bstats::constants::statistical::/stats::detail::/g;
        s/\bstats::constants::platform::/stats::detail::/g;
        s/\bstats::constants::thresholds::poisson::/stats::detail::/g;

        # Also handle cases where constants:: is used within stats namespace context
        s/\bconstants::math::/detail::/g;
        s/\bconstants::probability::/detail::/g;
        s/\bconstants::precision::/detail::/g;
        s/\bconstants::benchmark::/detail::/g;
        s/\bconstants::thresholds::/detail::/g;
        s/\bconstants::chi_squared::/detail::/g;
        s/\bconstants::goodness_of_fit::/detail::/g;
        s/\bconstants::robust::/detail::/g;
        s/\bconstants::statistical::/detail::/g;
        s/\bconstants::platform::/detail::/g;
        s/\bconstants::thresholds::poisson::/detail::/g;

        # Handle detail:: prefixed references that may have been partially updated
        s/\bdetail::math::/detail::/g;
        s/\bdetail::probability::/detail::/g;
        s/\bdetail::precision::/detail::/g;
        s/\bdetail::benchmark::/detail::/g;
        s/\bdetail::thresholds::/detail::/g;
        s/\bdetail::chi_squared::/detail::/g;
        s/\bdetail::goodness_of_fit::/detail::/g;
        s/\bdetail::robust::/detail::/g;
        s/\bdetail::statistical::/detail::/g;
        s/\bdetail::platform::/detail::/g;
        s/\bdetail::poisson::/detail::/g;
    ' "$file"
done

echo ""
echo "=== Consolidation complete! ==="
echo ""
echo "Summary of changes:"
echo "1. All constants moved from stats::constants::* to stats::detail::"
echo "2. Removed nested namespaces (math, probability, benchmark, thresholds, etc.)"
echo "3. Updated all references throughout the codebase"
echo ""
echo "Next steps:"
echo "1. Review the changes"
echo "2. Compile the project to check for any remaining issues"
echo "3. Run tests to ensure functionality is preserved"

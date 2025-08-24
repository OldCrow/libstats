#!/bin/bash

# Script to fix all math:: namespace references to detail:: in test files
# This is needed because we moved functions from stats::math to stats::detail

echo "Fixing math:: namespace references to detail:: in test files..."

# Find all test .cpp files and replace stats::math with stats::detail
find /Users/wolfman/Development/libstats/tests -name "*.cpp" -type f | while read file; do
    echo "Processing $file..."
    # Use sed to replace stats::math with stats::detail
    sed -i.bak 's/stats::math/stats::detail/g' "$file"
    # Also fix "using namespace stats::math" to use detail
    sed -i.bak2 's/using namespace stats::math/using namespace stats::detail/g' "$file"
    # Fix standalone math:: references to detail::
    sed -i.bak3 's/\bmath::/detail::/g' "$file"
done

echo "Cleaning up backup files..."
find /Users/wolfman/Development/libstats/tests -name "*.cpp.bak*" -type f -delete

echo "Done! All test file math:: references have been changed to detail::"

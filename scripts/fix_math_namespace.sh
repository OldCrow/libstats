#!/bin/bash

# Script to fix all math:: namespace references to detail::
# This is needed because we moved functions from stats::math to stats::detail

echo "Fixing math:: namespace references to detail:: in source files..."

# Find all .cpp files and replace math:: with detail::
find /Users/wolfman/Development/libstats/src -name "*.cpp" -type f | while read file; do
    echo "Processing $file..."
    # Use sed to replace math:: with detail::
    # Backup original files with .bak extension
    sed -i.bak 's/math::/detail::/g' "$file"
done

echo "Cleaning up backup files..."
find /Users/wolfman/Development/libstats/src -name "*.cpp.bak" -type f -delete

echo "Done! All math:: references have been changed to detail::"

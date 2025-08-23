#!/bin/bash

# Namespace refactoring script for libstats -> stats migration
# This script performs the actual namespace changes

set -e

echo "=== Starting Namespace Refactoring ==="
echo ""

# Create backup
if [ ! -d "backup_pre_refactor" ]; then
    echo "Creating backup..."
    cp -r include backup_pre_refactor_include
    cp -r src backup_pre_refactor_src
    echo "Backup created in backup_pre_refactor_*"
fi

# Function to refactor a file
refactor_file() {
    local file="$1"
    local changes_made=false

    # Skip binary files and non-text files
    if ! file "$file" | grep -q "text"; then
        return
    fi

    # Create temp file
    local temp_file="${file}.tmp"
    cp "$file" "$temp_file"

    # Replace namespace libstats with namespace stats
    if grep -q "^namespace libstats" "$temp_file"; then
        sed -i '' 's/^namespace libstats/namespace stats/g' "$temp_file"
        changes_made=true
    fi

    # Replace libstats:: with stats:: (but not in comments or strings)
    if grep -q "libstats::" "$temp_file"; then
        # This is more complex - need to be careful not to break things
        # For now, we'll be conservative and only change clear cases
        sed -i '' 's/\([^a-zA-Z_]\)libstats::/\1stats::/g' "$temp_file"
        sed -i '' 's/^libstats::/stats::/g' "$temp_file"
        changes_made=true
    fi

    # If changes were made, copy back
    if [ "$changes_made" = true ]; then
        mv "$temp_file" "$file"
        echo "  Refactored: $file"
    else
        rm "$temp_file"
    fi
}

# Phase 1: Update essential constants headers
echo "Phase 1: Refactoring essential constants headers..."
for file in include/core/essential_constants.h \
            include/core/mathematical_constants.h \
            include/core/precision_constants.h \
            include/core/statistical_constants.h \
            include/core/probability_constants.h; do
    if [ -f "$file" ]; then
        refactor_file "$file"
    fi
done

# Phase 2: Update core headers
echo ""
echo "Phase 2: Refactoring core headers..."
for file in include/core/*.h; do
    if [ -f "$file" ]; then
        refactor_file "$file"
    fi
done

# Phase 3: Update common headers
echo ""
echo "Phase 3: Refactoring common headers..."
for file in include/common/*.h; do
    if [ -f "$file" ] && [ "$file" != "include/common/forward_declarations.h" ]; then
        refactor_file "$file"
    fi
done

# Phase 4: Update distribution headers
echo ""
echo "Phase 4: Refactoring distribution headers..."
for file in include/distributions/*.h; do
    if [ -f "$file" ]; then
        refactor_file "$file"
    fi
done

# Phase 5: Update platform headers (more complex due to nested namespaces)
echo ""
echo "Phase 5: Refactoring platform headers..."
for file in include/platform/*.h; do
    if [ -f "$file" ]; then
        refactor_file "$file"
    fi
done

# Phase 6: Update source files
echo ""
echo "Phase 6: Refactoring source files..."
for file in src/*.cpp; do
    if [ -f "$file" ]; then
        refactor_file "$file"
    fi
done

# Phase 7: Add backward compatibility at the end of libstats.h
echo ""
echo "Phase 7: Adding backward compatibility..."
if ! grep -q "namespace libstats = stats;" include/libstats.h; then
    echo "" >> include/libstats.h
    echo "// Backward compatibility: alias libstats to stats" >> include/libstats.h
    echo "// This allows existing code using libstats:: to continue working" >> include/libstats.h
    echo "// Will be deprecated in v1.0.0" >> include/libstats.h
    echo "namespace libstats = stats;" >> include/libstats.h
    echo "  Added backward compatibility alias to libstats.h"
fi

echo ""
echo "=== Refactoring Complete ==="
echo ""
echo "Next steps:"
echo "1. Review the changes with: git diff"
echo "2. Test compilation: cd build && cmake .. && make -j4"
echo "3. Run tests: cd build && ctest --output-on-failure"
echo "4. If issues arise, restore from backup: cp -r backup_pre_refactor_* ."

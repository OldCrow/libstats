#!/bin/bash

# Check for copyright headers in source files
# Can be customized based on project requirements

EXIT_CODE=0
# CURRENT_YEAR=$(date +%Y)  # Reserved for future use

for file in "$@"; do
    # Skip if file doesn't exist
    [ -f "$file" ] || continue

    # Skip binary files
    if file "$file" | grep -q "binary"; then
        continue
    fi

    # Skip certain file types that don't need copyright
    case "$file" in
        *.md|*.txt|*.json|*.yaml|*.yml|.gitignore|.clang*|.cmake*|LICENSE|README)
            continue
            ;;
    esac

    # Check for copyright notice (customize this pattern as needed)
    if ! head -n 10 "$file" | grep -qE "(Copyright|copyright|©).*[0-9]{4}"; then
        echo "WARNING: $file may be missing copyright header"
        # For now, just warn - don't fail the commit
        # EXIT_CODE=1
    fi
done

exit $EXIT_CODE

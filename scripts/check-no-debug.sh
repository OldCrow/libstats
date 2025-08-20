#!/bin/bash

# Check for debug code that shouldn't be committed
# Looks for common debug patterns

EXIT_CODE=0

# Patterns to check for
DEBUG_PATTERNS=(
    "std::cout.*<<.*DEBUG"
    "printf.*DEBUG"
    "console\.log"
    "#define\s+DEBUG\s+1"
    "// *TODO:.*remove"
    "// *FIXME:.*temporary"
    "// *HACK:"
    "// *XXX:"
    "DO NOT COMMIT"
    "DO NOT SUBMIT"
)

for file in "$@"; do
    # Skip if file doesn't exist
    [ -f "$file" ] || continue

    # Check each pattern
    for pattern in "${DEBUG_PATTERNS[@]}"; do
        if grep -qE "$pattern" "$file"; then
            matches=$(grep -nE "$pattern" "$file")
            echo "WARNING: $file contains potential debug code:"
            echo "$matches"
            echo ""
            # For now, just warn - can change to EXIT_CODE=1 to fail
        fi
    done

    # Check for std::cout in non-test, non-example, non-tool files
    if [[ ! "$file" =~ (test|example|tool|benchmark) ]]; then
        if grep -q "std::cout" "$file"; then
            echo "WARNING: $file contains std::cout (should use proper logging)"
            echo "  Found at: $(grep -n 'std::cout' "$file" | head -3)"
            # EXIT_CODE=1  # Uncomment to make this a hard failure
        fi
    fi
done

exit $EXIT_CODE

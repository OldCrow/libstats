#!/bin/bash

# Validate include order in C++ files
# Expected order:
# 1. Corresponding header (for .cpp files)
# 2. C system headers
# 3. C++ standard library headers
# 4. Other library headers
# 5. Project headers

EXIT_CODE=0

check_include_order() {
    local file=$1
    local has_error=0

    # Extract all include lines with line numbers
    local includes
    includes=$(grep -n "^#include" "$file" 2>/dev/null)

    if [ -z "$includes" ]; then
        return 0
    fi

    # Track the last seen category
    # 0=none, 1=own header, 2=c headers, 3=c++ headers, 4=other libs, 5=project
    local last_category=0
    local line_num=0

    while IFS= read -r line; do
        line_num=$(echo "$line" | cut -d: -f1)
        include_line=$(echo "$line" | cut -d: -f2-)

        # Determine category
        if [[ "$include_line" =~ \#include\ \".*\.h\" ]] && [ "$last_category" -eq 0 ]; then
            # First include, likely corresponding header
            category=1
        elif [[ "$include_line" =~ \#include\ \<c[a-z]+\> ]]; then
            # C header like <cstdio>, <cmath>
            category=2
        elif [[ "$include_line" =~ \#include\ \<[a-z_]+\.h\> ]]; then
            # C system header like <stdio.h>
            category=2
        elif [[ "$include_line" =~ \#include\ \<[a-z_]+\> ]]; then
            # C++ standard library like <vector>, <iostream>
            category=3
        elif [[ "$include_line" =~ \#include\ \<.*\> ]]; then
            # Other library headers
            category=4
        else
            # Project headers with quotes
            category=5
        fi

        # Check if order is violated
        if [ "$category" -lt "$last_category" ] && [ "$category" -ne 1 ]; then
            echo "WARNING: $file:$line_num - Include order violation"
            echo "  Expected category order: own-header, C, C++, libraries, project"
            echo "  Line: $include_line"
            has_error=1
        fi

        # Update last category if not own header (own header is special)
        if [ "$category" -ne 1 ] || [ "$last_category" -eq 0 ]; then
            last_category=$category
        fi
    done <<< "$includes"

    return $has_error
}

for file in "$@"; do
    # Skip if file doesn't exist
    [ -f "$file" ] || continue

    # Only check .cpp and .cc files (headers have different rules)
    case "$file" in
        *.cpp|*.cc|*.cxx)
            if ! check_include_order "$file"; then
                # For now, just warn
                # EXIT_CODE=1
                :
            fi
            ;;
    esac
done

exit $EXIT_CODE

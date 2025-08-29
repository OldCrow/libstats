#!/bin/bash

echo "=== Analyzing Namespace Hierarchy ==="
echo

# Find all namespace declarations and their context
echo "=== Direct children of stats:: namespace ==="

# Look for patterns like "namespace stats { namespace X {" or standalone "namespace X {" within stats context
for file in $(find /Users/wolfman/Development/libstats -type f \( -name "*.cpp" -o -name "*.h" \) -not -path "*/build/*" -not -path "*/CMakeFiles/*"); do
    # Check if file contains namespace stats
    if grep -q "^namespace stats {" "$file"; then
        # Find namespaces declared directly within stats namespace
        awk '
        /^namespace stats \{/ { in_stats=1; next }
        /^namespace [a-zA-Z_][a-zA-Z0-9_]* \{/ && in_stats==1 {
            match($0, /namespace ([a-zA-Z_][a-zA-Z0-9_]*)/, arr)
            if (arr[1] != "stats") print arr[1]
        }
        /^\}  \/\/ namespace stats/ { in_stats=0 }
        ' "$file"
    fi
done | sort | uniq | while read ns; do
    count=$(grep -r "namespace $ns {" /Users/wolfman/Development/libstats --include="*.cpp" --include="*.h" | wc -l)
    echo "  $ns (found in $count places)"
done

echo
echo "=== Checking for libstats:: references (excluding comments) ==="
grep -r "libstats::" /Users/wolfman/Development/libstats --include="*.cpp" --include="*.h" | grep -v "//" | grep -v "CMakeFiles" | grep -v "build/" || echo "  None found"

echo
echo "=== Namespace alias for libstats ==="
grep -r "namespace libstats" /Users/wolfman/Development/libstats --include="*.cpp" --include="*.h" | grep -v "//" | grep -v "CMakeFiles" | grep -v "build/"

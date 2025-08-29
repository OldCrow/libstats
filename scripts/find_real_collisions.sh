#!/bin/bash

echo "=== Finding Real Namespace Collisions ==="
echo ""
echo "These constants/types will collide when namespaces are merged:"
echo ""

# Find duplicate constant names across different files/namespaces
echo "DUPLICATE CONSTANTS:"
echo "-------------------"
grep -h "constexpr.*[A-Z_][A-Z0-9_]*.*=" include/**/*.h 2>/dev/null | \
    sed -n 's/.*constexpr.*\s\+\([A-Z_][A-Z0-9_]*\)\s*=.*/\1/p' | \
    sort | uniq -c | sort -rn | \
    awk '$1 > 1 {print $2 " (defined " $1 " times)"}' | head -30

echo ""
echo "LOCATIONS OF KEY DUPLICATES:"
echo "-----------------------------"
for const in "LOG_PROBABILITY_EPSILON" "DEFAULT_BLOCK_SIZE" "MAX_BLOCK_SIZE" "MIN_ELEMENTS_FOR_PARALLEL" "SIMPLE_OPERATION_GRAIN_SIZE"; do
    count=$(grep -r "constexpr.*$const" include/ --include="*.h" 2>/dev/null | wc -l)
    if [ "$count" -gt 1 ]; then
        echo ""
        echo "$const appears $count times in:"
        grep -r "constexpr.*$const" include/ --include="*.h" 2>/dev/null | \
            sed 's/:.*namespace \([a-zA-Z_:]*\).*/: namespace \1/' | \
            sed 's/:.*$//' | sort | uniq
    fi
done

echo ""
echo "PLATFORM-SPECIFIC CONSTANTS (will need prefixing):"
echo "---------------------------------------------------"
grep -r "namespace \(sse\|avx\|avx2\|avx512\|neon\|intel\|amd\|arm\|apple_silicon\)" include/ --include="*.h" | \
    cut -d: -f1 | sort | uniq

echo ""
echo "FILES WITH MOST NAMESPACES (refactor first):"
echo "---------------------------------------------"
for file in include/**/*.h; do
    if [ -f "$file" ]; then
        NS_COUNT=$(grep -c "^namespace " "$file" 2>/dev/null || echo 0)
        if [ "$NS_COUNT" -gt 2 ]; then
            echo "$NS_COUNT namespaces: $file"
        fi
    fi
done | sort -rn | head -10

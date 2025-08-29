#!/bin/bash

# Detect potential namespace collisions in libstats
# This script identifies symbols that may conflict when namespaces are consolidated

echo "=== Detecting Potential Namespace Collisions ==="
echo ""

# Create temp directory for analysis
TEMP_DIR=$(mktemp -d)
trap 'rm -rf $TEMP_DIR' EXIT

# Find all constants
echo "Analyzing constants..."
grep -h "constexpr" include/**/*.h src/*.cpp 2>/dev/null | \
    grep -v "^//" | \
    grep -v "^\s*//" | \
    sed -n 's/.*constexpr.*\s\+\([A-Z_][A-Z0-9_]*\)\s*=.*/\1/p' | \
    sort | uniq -c | sort -rn | \
    awk '$1 > 1 {print $2 " (appears " $1 " times)"}' > "$TEMP_DIR/duplicate_constants.txt"

# Find all class/struct names
echo "Analyzing classes and structs..."
grep -h "^\s*\(class\|struct\)" include/**/*.h 2>/dev/null | \
    grep -v "^//" | \
    sed -n 's/.*\(class\|struct\)\s\+\([A-Za-z_][A-Za-z0-9_]*\).*/\2/p' | \
    sort | uniq -c | sort -rn | \
    awk '$1 > 1 {print $2 " (appears " $1 " times)"}' > "$TEMP_DIR/duplicate_types.txt"

# Find all function names (approximate)
echo "Analyzing functions..."
grep -h "^\s*\(inline\|static\|constexpr\)*\s*\w\+\s\+\w\+\s*(" include/**/*.h 2>/dev/null | \
    grep -v "^//" | \
    grep -v "^\s*//" | \
    sed -n 's/.*\s\+\([a-z_][a-zA-Z0-9_]*\)\s*(.*/\1/p' | \
    sort | uniq -c | sort -rn | \
    awk '$1 > 1 {print $2 " (appears " $1 " times)"}' > "$TEMP_DIR/duplicate_functions.txt"

# Find all enum names
echo "Analyzing enums..."
grep -h "^\s*enum" include/**/*.h 2>/dev/null | \
    grep -v "^//" | \
    sed -n 's/.*enum\s\+\(class\s\+\)*\([A-Za-z_][A-Za-z0-9_]*\).*/\2/p' | \
    sort | uniq -c | sort -rn | \
    awk '$1 > 1 {print $2 " (appears " $1 " times)"}' > "$TEMP_DIR/duplicate_enums.txt"

# Report findings
echo ""
echo "=== Collision Report ==="
echo ""

echo "Duplicate Constants:"
if [ -s "$TEMP_DIR/duplicate_constants.txt" ]; then
    head -20 "$TEMP_DIR/duplicate_constants.txt"
    CONST_COUNT=$(wc -l < "$TEMP_DIR/duplicate_constants.txt")
    echo "  ... Total: $CONST_COUNT duplicate constant names"
else
    echo "  None found"
fi
echo ""

echo "Duplicate Types (Classes/Structs):"
if [ -s "$TEMP_DIR/duplicate_types.txt" ]; then
    cat "$TEMP_DIR/duplicate_types.txt"
else
    echo "  None found"
fi
echo ""

echo "Duplicate Functions (approximate):"
if [ -s "$TEMP_DIR/duplicate_functions.txt" ]; then
    head -20 "$TEMP_DIR/duplicate_functions.txt"
    FUNC_COUNT=$(wc -l < "$TEMP_DIR/duplicate_functions.txt")
    echo "  ... Total: $FUNC_COUNT duplicate function names"
else
    echo "  None found"
fi
echo ""

echo "Duplicate Enums:"
if [ -s "$TEMP_DIR/duplicate_enums.txt" ]; then
    cat "$TEMP_DIR/duplicate_enums.txt"
else
    echo "  None found"
fi
echo ""

# Analyze namespace depth
echo "=== Namespace Depth Analysis ==="
grep -h "namespace" include/**/*.h 2>/dev/null | \
    grep -v "^//" | \
    grep -v "using namespace" | \
    sed 's/[^:]//g' | \
    awk '{print length}' | \
    sort -n | uniq -c | \
    awk '{
        if ($2 == 0) print "  Top-level namespaces: " $1
        else print "  Depth " $2 " nesting: " $1 " occurrences"
    }'

echo ""
echo "=== Files to Refactor (by namespace complexity) ==="
for file in include/**/*.h; do
    if [ -f "$file" ]; then
        NS_COUNT=$(grep -c "^namespace " "$file" 2>/dev/null || echo 0)
        if [ "$NS_COUNT" -gt 0 ]; then
            echo "$NS_COUNT namespaces: $file"
        fi
    fi
done | sort -rn | head -20

echo ""
echo "Analysis complete. Review the above before starting refactoring."

#!/bin/bash
# Local linting script for libstats
# Run this before committing to check code quality

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

echo "===================="
echo "LibStats Code Linter"
echo "===================="
echo ""

# Check if tools are available
CLANG_FORMAT=""
CLANG_TIDY=""

# Find clang-format (prefer version 15, but accept others)
for cmd in clang-format-15 clang-format-14 clang-format; do
    if command -v $cmd &> /dev/null; then
        CLANG_FORMAT=$cmd
        break
    fi
done

# Find clang-tidy (prefer version 15, but accept others)
for cmd in clang-tidy-15 clang-tidy-14 clang-tidy; do
    if command -v $cmd &> /dev/null; then
        CLANG_TIDY=$cmd
        break
    fi
done

# Check formatting
if [ -n "$CLANG_FORMAT" ]; then
    echo "🔍 Checking code formatting with $CLANG_FORMAT..."

    # Find all source files
    FILES=$(find include src tests -name '*.h' -o -name '*.cpp' 2>/dev/null || true)

    if [ -n "$FILES" ]; then
        FORMAT_ISSUES=0
        for file in $FILES; do
            if ! $CLANG_FORMAT --dry-run --Werror "$file" 2>/dev/null; then
                echo "  ❌ $file needs formatting"
                FORMAT_ISSUES=$((FORMAT_ISSUES + 1))
            fi
        done

        if [ $FORMAT_ISSUES -eq 0 ]; then
            echo "  ✅ All files are properly formatted"
        else
            echo ""
            echo "  💡 Run 'scripts/format.sh' to auto-format all files"
        fi
    fi
else
    echo "⚠️  clang-format not found - skipping format check"
    echo "    Install with: apt-get install clang-format-15 (or brew install clang-format on macOS)"
fi

echo ""

# Run clang-tidy
if [ -n "$CLANG_TIDY" ]; then
    echo "🔍 Running static analysis with $CLANG_TIDY..."

    # Need compile_commands.json
    if [ ! -f "build/compile_commands.json" ]; then
        echo "  Building compile_commands.json..."
        cmake -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON > /dev/null 2>&1
    fi

    # Check a subset of files (to keep it fast for local runs)
    echo "  Checking core source files..."

    # Find source files, limiting to core functionality for speed
    CORE_FILES=$(find src -name '*.cpp' -path '*/core/*' -o -path '*/distributions/*' 2>/dev/null | head -10 || true)

    if [ -n "$CORE_FILES" ]; then
        TIDY_ISSUES=0
        for file in $CORE_FILES; do
            # Run clang-tidy with our config, suppressing system header warnings
            OUTPUT=$($CLANG_TIDY "$file" -p build \
                --header-filter='.*/(include|src)/.*\.h$' \
                --system-headers=false \
                --quiet 2>&1 || true)

            if [ -n "$OUTPUT" ]; then
                # Filter out non-error output
                FILTERED=$(echo "$OUTPUT" | grep -E "warning:|error:" || true)
                if [ -n "$FILTERED" ]; then
                    echo "  Issues in $file:"
                    echo "$FILTERED" | sed 's/^/    /'
                    TIDY_ISSUES=$((TIDY_ISSUES + 1))
                fi
            fi
        done

        if [ $TIDY_ISSUES -eq 0 ]; then
            echo "  ✅ No issues found in checked files"
        fi

        echo "  ℹ️  Run 'scripts/lint-all.sh' for comprehensive analysis"
    fi
else
    echo "⚠️  clang-tidy not found - skipping static analysis"
    echo "    Install with: apt-get install clang-tidy-15 (or brew install llvm on macOS)"
fi

echo ""

# Check for common issues
echo "🔍 Checking for common issues..."

# Check for tabs in source files
TAB_FILES=$(find include src tests \( -name '*.h' -o -name '*.cpp' \) -exec grep -l $'\t' {} \; 2>/dev/null || true)
if [ -n "$TAB_FILES" ]; then
    echo "  ⚠️  Files containing tabs (should use spaces):"
    echo "$TAB_FILES" | sed 's/^/    /'
else
    echo "  ✅ No tabs found in source files"
fi

# Check for trailing whitespace
TRAILING_WS=$(find include src tests \( -name '*.h' -o -name '*.cpp' \) -exec grep -l '[[:space:]]$' {} \; 2>/dev/null || true)
if [ -n "$TRAILING_WS" ]; then
    echo "  ⚠️  Files with trailing whitespace:"
    echo "$TRAILING_WS" | head -5 | sed 's/^/    /'
    WS_COUNT=$(echo "$TRAILING_WS" | wc -l)
    if [ "$WS_COUNT" -gt 5 ]; then
        echo "    ... and $((WS_COUNT - 5)) more files"
    fi
else
    echo "  ✅ No trailing whitespace found"
fi

echo ""
echo "===================="
echo "Linting complete!"
echo "===================="

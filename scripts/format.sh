#!/bin/bash
# Auto-formatting script for libstats
# Formats all source files in the project

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

echo "========================"
echo "LibStats Code Formatter"
echo "========================"
echo ""

# Find clang-format command
CLANG_FORMAT=""
for cmd in clang-format-15 clang-format-14 clang-format; do
    if command -v $cmd &> /dev/null; then
        CLANG_FORMAT=$cmd
        break
    fi
done

if [ -z "$CLANG_FORMAT" ]; then
    echo "❌ Error: clang-format not found."
    echo "  Please install clang-format (e.g., 'sudo apt-get install clang-format-15' or 'brew install clang-format')"
    exit 1
fi

echo "🔍 Using $CLANG_FORMAT to format files..."

# Find all source and header files
FILES=$(find include src tests -name '*.h' -o -name '*.cpp' 2>/dev/null || true)

if [ -n "$FILES" ]; then
    echo "  Formatting the following files:"
    # Use xargs to handle a large number of files
    echo "$FILES" | xargs -n 1 -I {} echo "    - {}"

    # Run clang-format in-place
    echo "$FILES" | xargs -n 1 -I {} $CLANG_FORMAT -i "{}"

    echo ""
    echo "✅ Formatting complete!"
else
    echo "  No source files found to format."
fi

echo ""
echo "========================"

#!/bin/bash

# Include What You Use (IWYU) analysis script for libstats
# This script runs IWYU to check header dependencies and suggest improvements

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
BUILD_DIR="${BUILD_DIR:-build}"
IWYU_TOOL="iwyu_tool.py"
IWYU_FIX="fix_includes.py"
# MAPPING_FILE=".iwyu_mappings.imp"  # Reserved for future use
OUTPUT_FILE="iwyu_report.txt"

# Function to print colored messages
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Setup PATH for macOS with Homebrew LLVM
setup_macos_paths() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # Check for Intel Mac Homebrew path
        if [ -d "/usr/local/opt/llvm/bin" ] && [[ ":$PATH:" != *":/usr/local/opt/llvm/bin:"* ]]; then
            export PATH="/usr/local/opt/llvm/bin:$PATH"
            print_info "Added Intel Mac Homebrew LLVM to PATH"
        fi

        # Check for Apple Silicon Mac Homebrew path
        if [ -d "/opt/homebrew/opt/llvm/bin" ] && [[ ":$PATH:" != *":/opt/homebrew/opt/llvm/bin:"* ]]; then
            export PATH="/opt/homebrew/opt/llvm/bin:$PATH"
            print_info "Added Apple Silicon Mac Homebrew LLVM to PATH"
        fi
    fi
}

# Check if IWYU is installed
check_iwyu() {
    if ! command -v include-what-you-use &> /dev/null; then
        print_error "include-what-you-use is not installed!"
        echo "Install it using one of the following methods:"
        echo "  macOS: brew install include-what-you-use"
        echo "  Ubuntu: apt-get install iwyu"
        echo "  From source: https://github.com/include-what-you-use/include-what-you-use"
        exit 1
    fi

    # Check for IWYU tools - try different locations
    IWYU_TOOL_FOUND=""
    IWYU_FIX_FOUND=""

    # Common locations for IWYU tools (look for actual Python script, not shell wrapper)
    IWYU_SEARCH_PATHS=(
        "/usr/local/Cellar/include-what-you-use/*/libexec/bin/iwyu_tool.py"
        "/opt/homebrew/Cellar/include-what-you-use/*/libexec/bin/iwyu_tool.py"
        "/usr/local/Cellar/include-what-you-use/*/bin/iwyu_tool.py"
        "/opt/homebrew/Cellar/include-what-you-use/*/bin/iwyu_tool.py"
        "/usr/local/bin/iwyu_tool.py"
        "/opt/homebrew/bin/iwyu_tool.py"
    )

    # Search for iwyu_tool.py
    for path_pattern in "${IWYU_SEARCH_PATHS[@]}"; do
        for path in $path_pattern; do
            if [ -f "$path" ]; then
                IWYU_TOOL_FOUND="$path"
                # Look for fix_includes.py in the same directory
                IWYU_FIX_CANDIDATE="$(dirname "$path")/fix_includes.py"
                if [ -f "$IWYU_FIX_CANDIDATE" ]; then
                    IWYU_FIX_FOUND="$IWYU_FIX_CANDIDATE"
                fi
                print_info "Found IWYU tools at $(dirname "$path")"
                break 2
            fi
        done
    done

    if [ -z "$IWYU_TOOL_FOUND" ]; then
        print_error "Could not find iwyu_tool.py"
        echo "Searched in:"
        for path in "${IWYU_SEARCH_PATHS[@]}"; do
            echo "  $path"
        done
        echo "Please install IWYU: brew install include-what-you-use"
        exit 1
    fi

    IWYU_TOOL="$IWYU_TOOL_FOUND"
    IWYU_FIX="$IWYU_FIX_FOUND"
}

# Function to generate compilation database if needed
generate_compile_commands() {
    if [ ! -f "${BUILD_DIR}/compile_commands.json" ]; then
        print_info "Generating compilation database..."
        cmake -B ${BUILD_DIR} -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
    else
        print_info "Using existing compilation database"
    fi
}

# Function to run IWYU on specific targets
run_iwyu_analysis() {
    local target=$1
    local pattern=$2

    print_info "Analyzing $target..."

    if [ -z "$pattern" ]; then
        # Run on entire project
        python3 "${IWYU_TOOL}" -p "${BUILD_DIR}" \
            2>&1 | tee "${OUTPUT_FILE}"
    else
        # Find files matching pattern and run IWYU on each
        local files
        files=$(find src include -name "$pattern" -type f 2>/dev/null)
        if [ -z "$files" ]; then
            print_warning "No files found matching pattern: $pattern"
            return 1
        fi

        print_info "Found files: $files"

        # Run IWYU on the found files (convert to absolute paths)
        for file in $files; do
            print_info "Analyzing $file..."
            # Convert to absolute path
            abs_file="$(realpath "$file")"
            python3 "${IWYU_TOOL}" -p "${BUILD_DIR}" \
                "$abs_file" \
                2>&1 | tee -a "${OUTPUT_FILE}"
        done
    fi
}

# Function to apply IWYU suggestions
apply_fixes() {
    if [ -f "${OUTPUT_FILE}" ]; then
        print_warning "Applying IWYU suggestions..."
        print_warning "This will modify your source files. Create a backup first!"

        read -p "Do you want to apply the fixes? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            python3 ${IWYU_FIX} < ${OUTPUT_FILE}
            print_info "Fixes applied. Please review the changes."
        else
            print_info "Fixes not applied. Review ${OUTPUT_FILE} for suggestions."
        fi
    else
        print_error "No IWYU output file found"
    fi
}

# Function to run IWYU on a single file
analyze_single_file() {
    local file=$1

    if [ ! -f "$file" ]; then
        print_error "File not found: $file"
        exit 1
    fi

    print_info "Analyzing single file: $file"

    # Extract compile command for this file
    local compile_cmd
    compile_cmd=$(python3 -c "
import json
import sys
with open('${BUILD_DIR}/compile_commands.json') as f:
    commands = json.load(f)
    for cmd in commands:
        if cmd['file'].endswith('${file}'):
            print(' '.join(cmd['command'].split()[1:]))
            sys.exit(0)
")

    if [ -z "$compile_cmd" ]; then
        print_error "Could not find compile command for $file"
        exit 1
    fi

    # Run IWYU on the file
    include-what-you-use \
        $compile_cmd 2>&1 | tee "${file}.iwyu"

    print_info "Results saved to ${file}.iwyu"
}

# Main script
main() {
    print_info "Starting Include What You Use analysis for libstats"

    # Setup macOS paths automatically
    setup_macos_paths

    # Parse command line arguments
    case "${1:-}" in
        --help|-h)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --all          Run IWYU on entire project"
            echo "  --src          Run IWYU on source files only"
            echo "  --headers      Run IWYU on header files only"
            echo "  --file FILE    Run IWYU on a specific file"
            echo "  --apply        Apply suggested fixes (use with caution)"
            echo "  --help         Show this help message"
            exit 0
            ;;
        --all)
            check_iwyu
            generate_compile_commands
            run_iwyu_analysis "entire project" ""
            ;;
        --src)
            check_iwyu
            generate_compile_commands
            run_iwyu_analysis "source files" "*.cpp"
            ;;
        --headers)
            check_iwyu
            generate_compile_commands
            run_iwyu_analysis "header files" "*.h"
            ;;
        --file)
            check_iwyu
            generate_compile_commands
            analyze_single_file "$2"
            ;;
        --apply)
            apply_fixes
            ;;
        *)
            check_iwyu
            generate_compile_commands
            run_iwyu_analysis "entire project" ""
            ;;
    esac

    print_info "IWYU analysis complete"
}

# Run main function
main "$@"

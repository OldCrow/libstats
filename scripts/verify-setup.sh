#!/bin/bash

# Verification script for libstats CI/CD setup
# This script checks that all required tools are properly installed and configured

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored messages
print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

# Check if running on macOS with Homebrew LLVM
setup_macos_paths() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        if [ -d "/usr/local/opt/llvm/bin" ]; then
            export PATH="/usr/local/opt/llvm/bin:$PATH"
            print_info "Added Homebrew LLVM to PATH"
        elif [ -d "/opt/homebrew/opt/llvm/bin" ]; then
            export PATH="/opt/homebrew/opt/llvm/bin:$PATH"
            print_info "Added Homebrew LLVM (Apple Silicon) to PATH"
        fi
    fi
}

# Function to check if a command exists and print version
check_tool() {
    local tool=$1
    local expected_name=$2

    if command -v "$tool" &> /dev/null; then
        local version
        version=$($tool --version 2>&1 | head -1)
        print_success "$expected_name found: $version"
        return 0
    else
        print_error "$expected_name not found ($tool command not available)"
        return 1
    fi
}

# Function to check file existence
check_file() {
    local file=$1
    local description=$2

    if [ -f "$file" ]; then
        print_success "$description exists: $file"
        return 0
    else
        print_error "$description missing: $file"
        return 1
    fi
}

# Main verification function
main() {
    echo "=== LibStats CI/CD Setup Verification ==="
    echo ""

    # Setup paths for macOS
    setup_macos_paths

    local all_good=true

    # Check core tools
    echo "Checking core development tools..."
    check_tool "pre-commit" "Pre-commit" || all_good=false
    check_tool "include-what-you-use" "Include What You Use (IWYU)" || all_good=false
    check_tool "clang-format" "clang-format" || all_good=false
    check_tool "shellcheck" "ShellCheck" || all_good=false

    echo ""

    # Check configuration files
    echo "Checking configuration files..."
    check_file ".pre-commit-config.yaml" "Pre-commit configuration" || all_good=false
    check_file ".iwyu_mappings.imp" "IWYU mappings" || all_good=false
    check_file ".clang-format" "clang-format configuration" || all_good=false
    check_file ".clang-tidy" "clang-tidy configuration" || all_good=false

    echo ""

    # Check scripts
    echo "Checking utility scripts..."
    check_file "scripts/run-iwyu.sh" "IWYU analysis script" || all_good=false
    check_file "scripts/setup-pre-commit.sh" "Pre-commit setup script" || all_good=false
    check_file "scripts/check-pragma-once.sh" "Pragma once checker" || all_good=false

    echo ""

    # Check Git hooks
    echo "Checking Git integration..."
    check_file ".git/hooks/pre-commit" "Git pre-commit hook" || all_good=false

    echo ""

    # Check build system
    echo "Checking build system..."
    check_file "build/compile_commands.json" "Compilation database" || all_good=false

    echo ""

    # Test basic functionality
    echo "Testing basic functionality..."

    # Test pre-commit on a simple file
    if command -v pre-commit &> /dev/null; then
        if pre-commit --version &> /dev/null; then
            print_success "Pre-commit is functional"
        else
            print_error "Pre-commit installed but not working properly"
            all_good=false
        fi
    fi

    # Test clang-format on a simple input
    if command -v clang-format &> /dev/null; then
        if echo "int main(){return 0;}" | clang-format --style=file -assume-filename=test.cpp &> /dev/null; then
            print_success "clang-format is functional"
        else
            print_error "clang-format installed but not working properly"
            all_good=false
        fi
    fi

    echo ""

    # Summary
    if [ "$all_good" = true ]; then
        print_success "All CI/CD tools are properly configured!"
        echo ""
        echo "Next steps:"
        echo "  • Run 'pre-commit run --all-files' to check all files"
        echo "  • Run './scripts/run-iwyu.sh --src' to analyze headers"
        echo "  • Commit some changes to test the hooks automatically"
        echo ""
        echo "For more information, see docs/CI_CD_SETUP.md"
    else
        print_error "Some tools are missing or misconfigured."
        echo ""
        echo "Common fixes:"
        echo "  • Install missing tools: brew install pre-commit include-what-you-use shellcheck"
        echo "  • Run pre-commit install to set up Git hooks"
        echo "  • Ensure LLVM tools are in PATH (on macOS: export PATH=\"/usr/local/opt/llvm/bin:\$PATH\")"
        echo "  • Generate compilation database: cmake -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
        echo ""
        echo "For detailed setup instructions, see docs/CI_CD_SETUP.md"
        exit 1
    fi
}

# Run main function
main

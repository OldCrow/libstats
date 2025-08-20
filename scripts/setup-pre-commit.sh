#!/bin/bash

# Setup script for pre-commit hooks in libstats
# This script installs pre-commit and configures the git hooks

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Check if Python is installed
check_python() {
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
    else
        print_error "Python is not installed!"
        echo "Please install Python 3.7+ to use pre-commit"
        exit 1
    fi

    print_info "Using Python: $($PYTHON_CMD --version)"
}

# Check if pip is installed
check_pip() {
    if ! $PYTHON_CMD -m pip --version &> /dev/null; then
        print_error "pip is not installed!"
        echo "Please install pip for Python"
        exit 1
    fi
}

# Install pre-commit
install_precommit() {
    if command -v pre-commit &> /dev/null; then
        print_info "pre-commit is already installed: $(pre-commit --version)"
    else
        print_step "Installing pre-commit..."
        $PYTHON_CMD -m pip install --user pre-commit

        # Check if installation succeeded
        if ! command -v pre-commit &> /dev/null; then
            print_warning "pre-commit installed but not in PATH"
            echo "You may need to add ~/.local/bin to your PATH"
            echo "Add this to your shell configuration:"
            echo '  export PATH="$HOME/.local/bin:$PATH"'

            # Try to use the installed version directly
            if [ -f "$HOME/.local/bin/pre-commit" ]; then
                PRE_COMMIT="$HOME/.local/bin/pre-commit"
                print_info "Using pre-commit at $PRE_COMMIT"
            else
                print_error "Could not find pre-commit after installation"
                exit 1
            fi
        else
            PRE_COMMIT="pre-commit"
        fi
    fi
}

# Install additional tools
install_additional_tools() {
    print_step "Checking for additional tools..."

    # Check for clang-format
    if ! command -v clang-format &> /dev/null; then
        print_warning "clang-format is not installed"
        echo "  Install with: brew install clang-format (macOS)"
        echo "              : apt-get install clang-format (Ubuntu)"
    else
        print_info "clang-format found: $(clang-format --version | head -1)"
    fi

    # Check for shellcheck
    if ! command -v shellcheck &> /dev/null; then
        print_warning "shellcheck is not installed"
        echo "  Install with: brew install shellcheck (macOS)"
        echo "              : apt-get install shellcheck (Ubuntu)"
    else
        print_info "shellcheck found: $(shellcheck --version | grep version:)"
    fi

    # Check for cmake-format
    if ! command -v cmake-format &> /dev/null; then
        print_warning "cmake-format is not installed"
        echo "  Install with: pip install cmake-format"
    else
        print_info "cmake-format found"
    fi
}

# Setup git hooks
setup_git_hooks() {
    print_step "Setting up git hooks..."

    # Install the git hook scripts
    ${PRE_COMMIT} install

    if [ $? -eq 0 ]; then
        print_info "Git hooks installed successfully!"
    else
        print_error "Failed to install git hooks"
        exit 1
    fi

    # Install hooks for commit messages (optional)
    ${PRE_COMMIT} install --hook-type commit-msg 2>/dev/null || true
}

# Run initial checks
run_initial_check() {
    print_step "Running initial pre-commit checks..."

    # Run on all files to see current status
    ${PRE_COMMIT} run --all-files || true

    print_info "Initial check complete. Fix any issues before committing."
}

# Create secrets baseline if it doesn't exist
setup_secrets_baseline() {
    if [ ! -f ".secrets.baseline" ]; then
        print_step "Creating secrets baseline..."
        if command -v detect-secrets &> /dev/null; then
            detect-secrets scan > .secrets.baseline
            print_info "Secrets baseline created"
        else
            print_warning "detect-secrets not installed, skipping baseline creation"
            echo "  Install with: pip install detect-secrets"
        fi
    fi
}

# Main execution
main() {
    print_info "Setting up pre-commit hooks for libstats"
    echo ""

    # Check prerequisites
    check_python
    check_pip

    # Install pre-commit
    install_precommit

    # Install additional tools
    install_additional_tools

    # Setup secrets baseline
    setup_secrets_baseline

    # Setup git hooks
    setup_git_hooks

    echo ""
    print_info "Setup complete! Pre-commit hooks are now active."
    print_info "Hooks will run automatically on 'git commit'"
    echo ""
    echo "Useful commands:"
    echo "  pre-commit run --all-files    # Run on all files"
    echo "  pre-commit run --files FILE   # Run on specific files"
    echo "  pre-commit autoupdate         # Update hook versions"
    echo "  git commit --no-verify        # Skip hooks (not recommended)"
    echo ""

    # Ask if user wants to run initial check
    read -p "Run initial check on all files? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        run_initial_check
    fi
}

# Run main function
main

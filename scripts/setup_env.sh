#!/bin/bash
# Environment setup for libstats development on Apple Silicon Mac

# Add Homebrew LLVM tools to PATH
export PATH="/opt/homebrew/opt/llvm/bin:$PATH"

echo "✓ Environment configured for libstats development"
echo "  - clang-format: $(which clang-format)"
echo "  - clang-tidy: $(which clang-tidy)"
echo "  - shellcheck: $(which shellcheck)"
echo ""
echo "Pre-commit hooks are installed and ready to use!"

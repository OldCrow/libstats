#!/bin/bash

# Phase 1 Execution Script: Magic Number Elimination
# This script automates the systematic replacement of magic numbers across the libstats codebase

set -e  # Exit on error

# Configuration
LIBSTATS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$LIBSTATS_ROOT/build"
SCRIPT_DIR="$LIBSTATS_ROOT/tools"
BACKUP_DIR="$LIBSTATS_ROOT/backups/phase1_$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LIBSTATS_ROOT/phase1_execution.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

# Print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

# File lists based on our analysis
HIGH_IMPACT_FILES=(
    "src/gamma.cpp"
    "src/math_utils.cpp"
    "src/poisson.cpp"
    "src/discrete.cpp"
    "src/validation.cpp"
    "src/uniform.cpp"
)

MEDIUM_LOW_IMPACT_FILES=(
    "src/distribution_base.cpp"
    "src/benchmark.cpp"
    "src/gaussian.cpp"
    "src/system_capabilities.cpp"
    "src/performance_dispatcher.cpp"
    "src/cpu_detection.cpp"
    "src/log_space_ops.cpp"
)

MINIMAL_IMPACT_FILES=(
    "src/performance_history.cpp"
    "src/simd_neon.cpp"
    "src/platform_constants_impl.cpp"
    "src/work_stealing_pool.cpp"
    "src/parallel_thresholds.cpp"
    "src/simd_fallback.cpp"
)

# Function to create backups
create_backup() {
    local file="$1"
    mkdir -p "$BACKUP_DIR/$(dirname "$file")"
    cp "$LIBSTATS_ROOT/$file" "$BACKUP_DIR/$file"
    print_status "Created backup for $file"
}

# Function to verify compilation
verify_compilation() {
    print_status "Verifying compilation..."
    cd "$BUILD_DIR"

    if make -j"$(nproc)" > /dev/null 2>&1; then
        print_success "Compilation successful"
        return 0
    else
        print_error "Compilation failed"
        return 1
    fi
}

# Function to process a single file
process_file() {
    local file="$1"
    local mode="$2"  # "interactive" or "batch"

    print_status "Processing $file..."

    # Create backup
    create_backup "$file"

    # Get suggestion count
    local suggestions
    suggestions=$(python3 "$SCRIPT_DIR/replace_magic_numbers.py" "$LIBSTATS_ROOT/$file" | grep "Suggest:" | wc -l | tr -d ' ')

    if [ "$suggestions" -eq 0 ]; then
        print_status "No suggestions for $file, skipping"
        return 0
    fi

    print_status "Found $suggestions suggestions for $file"

    # Apply replacements
    if [ "$mode" = "interactive" ]; then
        print_status "Running in interactive mode - please confirm each replacement"
        python3 "$SCRIPT_DIR/replace_magic_numbers.py" "$LIBSTATS_ROOT/$file" --interactive
    else
        python3 "$SCRIPT_DIR/replace_magic_numbers.py" "$LIBSTATS_ROOT/$file" --write
    fi

    # Verify compilation
    if ! verify_compilation; then
        print_error "Compilation failed after processing $file"
        print_warning "Restoring backup..."
        cp "$BACKUP_DIR/$file" "$LIBSTATS_ROOT/$file"
        verify_compilation
        return 1
    fi

    # Show diff
    print_status "Changes made to $file:"
    if command -v git &> /dev/null; then
        git --no-pager diff "$LIBSTATS_ROOT/$file" || true
    else
        diff "$BACKUP_DIR/$file" "$LIBSTATS_ROOT/$file" || true
    fi

    print_success "Successfully processed $file"
    return 0
}

# Function to commit changes
commit_changes() {
    local file="$1"
    local replacements="$2"

    if command -v git &> /dev/null && git rev-parse --git-dir > /dev/null 2>&1; then
        git add "$LIBSTATS_ROOT/$file"
        git commit -m "refactor: eliminate magic numbers in $(basename "$file")

Replace magic numbers with named constants from stats::detail namespace.
- Applied $replacements magic number replacements
- Enhanced code readability and maintainability

Phase 1 magic number elimination - automated processing"
        print_success "Committed changes for $file"
    else
        print_warning "Git not available or not in git repository - skipping commit"
    fi
}

# Main execution function
main() {
    print_status "Starting Phase 1 Magic Number Elimination"
    log "Execution started at $(date)"

    # Verify prerequisites
    print_status "Checking prerequisites..."

    if [ ! -f "$SCRIPT_DIR/replace_magic_numbers.py" ]; then
        print_error "Magic number replacement script not found!"
        exit 1
    fi

    if [ ! -d "$BUILD_DIR" ]; then
        print_error "Build directory not found! Please run cmake first."
        exit 1
    fi

    # Initial compilation check
    print_status "Verifying initial compilation state..."
    if ! verify_compilation; then
        print_error "Initial compilation failed - please fix build issues first"
        exit 1
    fi

    # Create backup directory
    mkdir -p "$BACKUP_DIR"
    print_status "Created backup directory: $BACKUP_DIR"

    local total_processed=0
    local total_failed=0

    # Process high-impact files (interactive mode)
    print_status "=== Phase 1A: Processing High-Impact Files (Interactive Mode) ==="
    for file in "${HIGH_IMPACT_FILES[@]}"; do
        if [ -f "$LIBSTATS_ROOT/$file" ]; then
            if process_file "$file" "interactive"; then
                ((total_processed++))
            else
                ((total_failed++))
            fi
        else
            print_warning "File $file not found, skipping"
        fi
    done

    # Ask if user wants to continue with batch processing
    echo
    read -p "Continue with batch processing of medium/low-impact files? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_status "Stopping execution as requested"
        print_status "Summary: $total_processed files processed, $total_failed files failed"
        exit 0
    fi

    # Process medium/low-impact files (batch mode)
    print_status "=== Phase 1B: Processing Medium/Low-Impact Files (Batch Mode) ==="
    for file in "${MEDIUM_LOW_IMPACT_FILES[@]}"; do
        if [ -f "$LIBSTATS_ROOT/$file" ]; then
            if process_file "$file" "batch"; then
                ((total_processed++))
            else
                ((total_failed++))
            fi
        else
            print_warning "File $file not found, skipping"
        fi
    done

    # Process minimal-impact files (batch mode)
    print_status "=== Phase 1C: Processing Minimal-Impact Files (Batch Mode) ==="
    for file in "${MINIMAL_IMPACT_FILES[@]}"; do
        if [ -f "$LIBSTATS_ROOT/$file" ]; then
            if process_file "$file" "batch"; then
                ((total_processed++))
            else
                ((total_failed++))
            fi
        else
            print_warning "File $file not found, skipping"
        fi
    done

    # Final summary
    print_status "=== Phase 1 Execution Complete ==="
    print_success "Successfully processed: $total_processed files"
    if [ "$total_failed" -gt 0 ]; then
        print_warning "Failed to process: $total_failed files"
    fi

    # Final compilation test
    print_status "Performing final compilation test..."
    if verify_compilation; then
        print_success "All files compile successfully!"
    else
        print_error "Final compilation failed - please review changes"
    fi

    print_status "Backups saved in: $BACKUP_DIR"
    print_status "Execution log saved in: $LOG_FILE"

    log "Phase 1 execution completed at $(date)"
}

# Help function
show_help() {
    echo "Phase 1 Magic Number Elimination Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help     Show this help message"
    echo "  --dry-run      Show what would be done without making changes"
    echo ""
    echo "This script systematically processes all source files to eliminate magic numbers"
    echo "by replacing them with named constants from the stats::detail namespace."
    echo ""
    echo "Processing order:"
    echo "  1. High-impact files (interactive mode)"
    echo "  2. Medium/low-impact files (batch mode)"
    echo "  3. Minimal-impact files (batch mode)"
    echo ""
    echo "Backups are automatically created before processing any file."
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Execute main function
if [ "${DRY_RUN:-0}" -eq 1 ]; then
    print_status "DRY RUN MODE - No changes will be made"
    # Show what would be processed
    echo "Would process the following files:"
    echo "High-impact files: ${HIGH_IMPACT_FILES[*]}"
    echo "Medium/low-impact files: ${MEDIUM_LOW_IMPACT_FILES[*]}"
    echo "Minimal-impact files: ${MINIMAL_IMPACT_FILES[*]}"
else
    main
fi

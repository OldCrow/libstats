#!/bin/bash

# Phase 2 Namespace Consolidation Script
# Goal: Reduce to 3-5 namespaces total as per NAMESPACE_PHASE2_PLAN.md

set -e

echo "=== Phase 2 Namespace Consolidation ==="
echo "Goal: stats::, stats::detail::, stats::test::, stats::arch:: (with subnamespaces)"
echo

# Create backup
BACKUP_DIR="backup_$(date +%Y%m%d_%H%M%S)"
echo "Creating backup in $BACKUP_DIR..."
mkdir -p "$BACKUP_DIR"
cp -r include "$BACKUP_DIR/"
cp -r src "$BACKUP_DIR/"
cp -r tests "$BACKUP_DIR/"
cp -r tools "$BACKUP_DIR/"
echo "Backup created."
echo

# Function to replace namespace declarations and usage
replace_namespace() {
    local old_ns="$1"
    local new_ns="$2"
    local file_pattern="${3:-*}"

    echo "  Replacing namespace $old_ns with $new_ns..."

    # Replace namespace declarations
    find . -type f \( -name "*.h" -o -name "*.cpp" \) -path "*/$file_pattern/*" 2>/dev/null | while read -r file; do
        # Skip backup directories
        [[ "$file" =~ backup_ ]] && continue

        # Replace namespace declarations
        sed -i '' "s/namespace $old_ns {/namespace $new_ns {/g" "$file" 2>/dev/null || true
        sed -i '' "s/}  \/\/ namespace $old_ns/}  \/\/ namespace $new_ns/g" "$file" 2>/dev/null || true

        # Replace namespace usage
        sed -i '' "s/using namespace stats::$old_ns;/using namespace stats::$new_ns;/g" "$file" 2>/dev/null || true
        sed -i '' "s/$old_ns::/detail::/g" "$file" 2>/dev/null || true
        sed -i '' "s/stats::$old_ns::/stats::$new_ns::/g" "$file" 2>/dev/null || true
    done
}

# Step 1: Move stats::safety to stats::detail
echo "Step 1: Moving stats::safety to stats::detail..."
find . -type f \( -name "*.h" -o -name "*.cpp" \) | while read -r file; do
    [[ "$file" =~ backup_ ]] && continue

    # In files where safety is a child of stats, move it to detail
    if grep -q "namespace safety {" "$file" 2>/dev/null; then
        # Check if it's within stats namespace
        if grep -B5 "namespace safety {" "$file" | grep -q "namespace stats {"; then
            sed -i '' '/namespace safety {/,/}  \/\/ namespace safety/ {
                s/namespace safety {/\/\/ Safety utilities moved to detail namespace/
                /}  \/\/ namespace safety/d
            }' "$file"
        fi
    fi

    # Update references
    sed -i '' 's/stats::safety::/stats::detail::/g' "$file" 2>/dev/null || true
    sed -i '' 's/using namespace stats::safety;/using namespace stats::detail;/g' "$file" 2>/dev/null || true
    sed -i '' 's/safety::/detail::/g' "$file" 2>/dev/null || true
done

# Step 2: Move stats::simd to stats::arch::simd
echo "Step 2: Moving stats::simd to stats::arch::simd..."
find . -type f \( -name "*.h" -o -name "*.cpp" \) | while read -r file; do
    [[ "$file" =~ backup_ ]] && continue

    # Update namespace declarations
    sed -i '' 's/^namespace simd {/namespace arch { namespace simd {/g' "$file" 2>/dev/null || true
    sed -i '' 's/^}  \/\/ namespace simd/} }  \/\/ namespace arch::simd/g' "$file" 2>/dev/null || true

    # Update references
    sed -i '' 's/stats::simd::/stats::arch::simd::/g' "$file" 2>/dev/null || true
    sed -i '' 's/using namespace stats::simd;/using namespace stats::arch::simd;/g' "$file" 2>/dev/null || true
done

# Step 3: Move stats::performance and nested namespaces to stats::detail
echo "Step 3: Moving stats::performance to stats::detail..."
find . -type f \( -name "*.h" -o -name "*.cpp" \) | while read -r file; do
    [[ "$file" =~ backup_ ]] && continue

    # First handle nested namespaces like performance::characteristics
    sed -i '' 's/stats::performance::characteristics::/stats::detail::/g' "$file" 2>/dev/null || true
    sed -i '' 's/performance::characteristics::/detail::/g' "$file" 2>/dev/null || true

    # Then handle main performance namespace
    sed -i '' 's/stats::performance::/stats::detail::/g' "$file" 2>/dev/null || true
    sed -i '' 's/using namespace stats::performance;/using namespace stats::detail;/g' "$file" 2>/dev/null || true
    sed -i '' 's/namespace performance {/namespace detail { \/\/ Performance utilities/g' "$file" 2>/dev/null || true
    sed -i '' 's/}  \/\/ namespace performance/}  \/\/ namespace detail/g' "$file" 2>/dev/null || true
done

# Step 4: Move other namespaces to stats::detail
echo "Step 4: Moving other namespaces to stats::detail..."
for ns in characteristics scaling adaptive validation; do
    echo "  Moving stats::$ns to stats::detail..."
    find . -type f \( -name "*.h" -o -name "*.cpp" \) | while read -r file; do
        [[ "$file" =~ backup_ ]] && continue

        sed -i '' "s/stats::$ns::/stats::detail::/g" "$file" 2>/dev/null || true
        sed -i '' "s/using namespace stats::$ns;/using namespace stats::detail;/g" "$file" 2>/dev/null || true
        sed -i '' "s/namespace $ns {/namespace detail { \/\/ $ns utilities/g" "$file" 2>/dev/null || true
        sed -i '' "s/}  \/\/ namespace $ns/}  \/\/ namespace detail/g" "$file" 2>/dev/null || true
    done
done

# Step 5: Move stats::tools and sub-namespaces to stats::detail
echo "Step 5: Moving stats::tools to stats::detail..."
for ns in tools time strings format table perf_utils display system_info tool_utils; do
    echo "  Moving $ns to detail..."
    find . -type f \( -name "*.h" -o -name "*.cpp" \) | while read -r file; do
        [[ "$file" =~ backup_ ]] && continue

        # Handle nested namespaces
        sed -i '' "s/stats::tools::$ns::/stats::detail::/g" "$file" 2>/dev/null || true
        sed -i '' "s/tools::$ns::/detail::/g" "$file" 2>/dev/null || true

        # Handle main namespace
        sed -i '' "s/stats::$ns::/stats::detail::/g" "$file" 2>/dev/null || true
        sed -i '' "s/using namespace stats::tools::$ns;/using namespace stats::detail;/g" "$file" 2>/dev/null || true
        sed -i '' "s/using namespace stats::$ns;/using namespace stats::detail;/g" "$file" 2>/dev/null || true
        sed -i '' "s/namespace $ns {/namespace detail { \/\/ $ns utilities/g" "$file" 2>/dev/null || true
        sed -i '' "s/}  \/\/ namespace $ns/}  \/\/ namespace detail/g" "$file" 2>/dev/null || true
    done
done

# Handle stats::tools itself
find . -type f \( -name "*.h" -o -name "*.cpp" \) | while read -r file; do
    [[ "$file" =~ backup_ ]] && continue
    sed -i '' "s/stats::tools::/stats::detail::/g" "$file" 2>/dev/null || true
    sed -i '' "s/using namespace stats::tools;/using namespace stats::detail;/g" "$file" 2>/dev/null || true
done

# Step 6: Move stats::testing to stats::test
echo "Step 6: Moving stats::testing to stats::test..."
find . -type f \( -name "*.h" -o -name "*.cpp" \) | while read -r file; do
    [[ "$file" =~ backup_ ]] && continue

    sed -i '' 's/stats::testing::/stats::test::/g' "$file" 2>/dev/null || true
    sed -i '' 's/using namespace stats::testing;/using namespace stats::test;/g' "$file" 2>/dev/null || true
    sed -i '' 's/namespace testing {/namespace test {/g' "$file" 2>/dev/null || true
    sed -i '' 's/}  \/\/ namespace testing/}  \/\/ namespace test/g' "$file" 2>/dev/null || true
done

echo
echo "=== Namespace Consolidation Complete ==="
echo
echo "Summary of changes:"
echo "  - stats::safety → stats::detail"
echo "  - stats::simd → stats::arch::simd"
echo "  - stats::performance → stats::detail"
echo "  - stats::characteristics → stats::detail"
echo "  - stats::scaling → stats::detail"
echo "  - stats::adaptive → stats::detail"
echo "  - stats::validation → stats::detail"
echo "  - stats::tools (and sub-namespaces) → stats::detail"
echo "  - stats::testing → stats::test"
echo
echo "Backup saved in: $BACKUP_DIR"
echo
echo "Next steps:"
echo "  1. cd build && make clean && make -j8"
echo "  2. Run tests to verify: ctest --output-on-failure"
echo "  3. Check for compilation errors and fix any remaining issues"

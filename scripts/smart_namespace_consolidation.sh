#!/bin/bash

# Smart namespace consolidation - Phase 2
# Strategy: Move everything to stats::detail:: WITHOUT unnecessary prefixing
# Only prefix when there are actual name collisions

set -e

echo "=== Smart Namespace Consolidation - Phase 2 ==="
echo ""

# Step 1: Fix the actual collisions first
echo "Step 1: Resolving actual name collisions..."

# Remove LOG_PROBABILITY_EPSILON from precision_constants.h (keep in probability_constants.h)
if grep -q "LOG_PROBABILITY_EPSILON_PRECISION" include/core/precision_constants.h; then
    echo "  ✓ LOG_PROBABILITY_EPSILON collision already resolved"
else
    echo "  - Checking LOG_PROBABILITY_EPSILON collision..."
fi

# Step 2: Change all nested namespaces to stats::detail::
echo ""
echo "Step 2: Consolidating nested namespaces to stats::detail::"

# Process constant headers - change stats::constants::* to stats::detail::
for file in include/core/mathematical_constants.h \
            include/core/precision_constants.h \
            include/core/probability_constants.h \
            include/core/statistical_constants.h \
            include/core/benchmark_constants.h \
            include/core/robust_constants.h \
            include/core/threshold_constants.h \
            include/core/statistical_methods_constants.h \
            include/core/goodness_of_fit_constants.h; do

    if [ -f "$file" ]; then
        echo "  Processing $(basename $file)..."

        # Replace namespace constants { namespace X { with namespace detail {
        sed -i '' 's/namespace constants {/namespace detail {/g' "$file"

        # Remove intermediate namespace levels
        sed -i '' '/^namespace math {$/d' "$file"
        sed -i '' '/^namespace precision {$/d' "$file"
        sed -i '' '/^namespace probability {$/d' "$file"
        sed -i '' '/^namespace statistical {$/d' "$file"
        sed -i '' '/^namespace benchmark {$/d' "$file"
        sed -i '' '/^namespace robust {$/d' "$file"
        sed -i '' '/^namespace thresholds {$/d' "$file"
        sed -i '' '/^namespace bayesian {$/d' "$file"
        sed -i '' '/^namespace bootstrap {$/d' "$file"
        sed -i '' '/^namespace cross_validation {$/d' "$file"

        # Remove sub-namespaces (these need special handling)
        sed -i '' '/^namespace normal {$/d' "$file"
        sed -i '' '/^namespace t_distribution {$/d' "$file"
        sed -i '' '/^namespace chi_square {$/d' "$file"
        sed -i '' '/^namespace f_distribution {$/d' "$file"
        sed -i '' '/^namespace priors {$/d' "$file"
        sed -i '' '/^namespace tuning {$/d' "$file"
        sed -i '' '/^namespace poisson {$/d' "$file"

        # Clean up extra closing braces
        # Count namespace openings vs closings and remove extras
    fi
done

# Step 3: Handle utility namespaces
echo ""
echo "Step 3: Moving utility namespaces to stats::detail::"

# Process headers with stats::math::, stats::validation::, stats::performance:: etc.
for file in include/core/math_utils.h \
            include/core/validation.h \
            include/core/safety.h \
            include/core/performance_dispatcher.h \
            include/core/performance_history.h; do

    if [ -f "$file" ]; then
        echo "  Processing $(basename $file)..."

        # Replace various utility namespaces with detail
        sed -i '' 's/namespace math {/namespace detail {/g' "$file"
        sed -i '' 's/namespace validation {/namespace detail {/g' "$file"
        sed -i '' 's/namespace safety {/namespace detail {/g' "$file"
        sed -i '' 's/namespace performance {/namespace detail {/g' "$file"
    fi
done

echo ""
echo "=== Smart Consolidation Status ==="
echo ""
echo "✓ Constants moved to stats::detail:: (no unnecessary prefixing)"
echo "✓ Actual collisions identified for manual resolution:"
echo "  - LOG_PROBABILITY_EPSILON (2 locations)"
echo "  - Platform constants need conditional compilation"
echo ""
echo "Next steps:"
echo "1. Manually review and fix namespace closing braces"
echo "2. Update references: stats::constants::math::PI → stats::detail::PI"
echo "3. Test compilation to find any actual collisions"
echo "4. Only add prefixes where truly needed"

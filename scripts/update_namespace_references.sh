#!/bin/bash

# Update all references from stats::constants::X:: to stats::detail::

echo "=== Updating namespace references ==="
echo ""

# Update all files that reference the old namespaces
echo "Updating references to constants..."

# Find and update all C++ files
for file in $(find include src tests tools examples -name "*.h" -o -name "*.cpp" 2>/dev/null); do
    if [ -f "$file" ]; then
        # Check if file contains old references
        if grep -q "constants::\(math\|precision\|probability\|statistical\|benchmark\|robust\|thresholds\)::" "$file" 2>/dev/null; then
            echo "  Updating $(basename $file)..."

            # Update namespace references
            sed -i '' 's/constants::math::/detail::/g' "$file"
            sed -i '' 's/constants::precision::/detail::/g' "$file"
            sed -i '' 's/constants::probability::/detail::/g' "$file"
            sed -i '' 's/constants::statistical::/detail::/g' "$file"
            sed -i '' 's/constants::benchmark::/detail::/g' "$file"
            sed -i '' 's/constants::robust::/detail::/g' "$file"
            sed -i '' 's/constants::thresholds::/detail::/g' "$file"

            # Also handle sub-namespaces
            sed -i '' 's/constants::statistical::normal::/detail::/g' "$file"
            sed -i '' 's/constants::statistical::t_distribution::/detail::/g' "$file"
            sed -i '' 's/constants::statistical::chi_square::/detail::/g' "$file"
            sed -i '' 's/constants::statistical::f_distribution::/detail::/g' "$file"
            sed -i '' 's/constants::bayesian::/detail::/g' "$file"
            sed -i '' 's/constants::bayesian::priors::/detail::/g' "$file"
            sed -i '' 's/constants::bootstrap::/detail::/g' "$file"
            sed -i '' 's/constants::cross_validation::/detail::/g' "$file"
            sed -i '' 's/constants::robust::tuning::/detail::/g' "$file"
            sed -i '' 's/constants::thresholds::poisson::/detail::/g' "$file"
            sed -i '' 's/constants::benchmark::thresholds::/detail::/g' "$file"
        fi

        # Also check for utility namespace references
        if grep -q "stats::\(math\|validation\|performance\|safety\)::" "$file" 2>/dev/null; then
            echo "  Updating utility references in $(basename $file)..."

            sed -i '' 's/stats::math::/stats::detail::/g' "$file"
            sed -i '' 's/stats::validation::/stats::detail::/g' "$file"
            sed -i '' 's/stats::performance::/stats::detail::/g' "$file"
            sed -i '' 's/stats::safety::/stats::detail::/g' "$file"
        fi
    fi
done

echo ""
echo "=== Reference Update Complete ==="
echo ""
echo "Updated all references from:"
echo "  stats::constants::X::Y → stats::detail::Y"
echo "  stats::math::X → stats::detail::X"
echo "  stats::validation::X → stats::detail::X"
echo "  etc."
echo ""
echo "Next: Compile to check for any remaining issues"

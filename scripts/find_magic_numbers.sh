#!/bin/bash

# find_magic_numbers.sh - Detect magic numbers in libstats source code
# This script finds numeric literals that should potentially be replaced with named constants

echo "======================================================================="
echo "               Magic Number Detection for libstats                    "
echo "======================================================================="
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Output file for detailed results
OUTPUT_FILE="magic_numbers_report.txt"
CSV_FILE="magic_numbers.csv"

# Initialize files
echo "Magic Numbers Report - Generated $(date)" > "$OUTPUT_FILE"
echo "=======================================" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

echo "File,Line,Value,Context,Category" > "$CSV_FILE"

# Function to check if a line is in a comment or string
is_comment_or_string() {
    local line="$1"
    # Check for comment markers
    if [[ "$line" =~ ^[[:space:]]*/\* ]] || [[ "$line" =~ ^[[:space:]]*\* ]] || [[ "$line" =~ ^[[:space:]]*// ]]; then
        return 0
    fi
    # Check if the number appears within quotes
    if [[ "$line" =~ \".*$2.*\" ]]; then
        return 0
    fi
    return 1
}

# Function to categorize magic numbers
categorize_number() {
    local value="$1"

    # Statistical critical values
    if [[ "$value" == "3.841" ]] || [[ "$value" == "5.991" ]] || [[ "$value" == "7.815" ]] || \
       [[ "$value" == "9.488" ]] || [[ "$value" == "11.070" ]]; then
        echo "chi_squared_critical"
        return
    fi

    # Normal quantiles
    if [[ "$value" == "1.645" ]] || [[ "$value" == "1.96" ]] || [[ "$value" == "2.576" ]]; then
        echo "normal_quantile"
        return
    fi

    # Significance levels
    if [[ "$value" == "0.01" ]] || [[ "$value" == "0.05" ]] || [[ "$value" == "0.10" ]] || \
       [[ "$value" == "0.90" ]] || [[ "$value" == "0.95" ]] || [[ "$value" == "0.99" ]]; then
        echo "significance_level"
        return
    fi

    # Mathematical constants
    if [[ "$value" == "0.5" ]] || [[ "$value" == "-0.5" ]] || [[ "$value" == "2.0" ]] || \
       [[ "$value" == "3.0" ]] || [[ "$value" == "6.0" ]] || [[ "$value" == "9.0" ]]; then
        echo "mathematical"
        return
    fi

    # Convergence tolerances
    if [[ "$value" =~ ^1e-[0-9]+$ ]] || [[ "$value" =~ ^1\.[0-9]*e-[0-9]+$ ]]; then
        echo "convergence_tolerance"
        return
    fi

    # Iteration limits
    if [[ "$value" == "100" ]] || [[ "$value" == "200" ]] || [[ "$value" == "500" ]] || \
       [[ "$value" == "1000" ]] || [[ "$value" == "5000" ]]; then
        echo "iteration_limit"
        return
    fi

    echo "uncategorized"
}

echo -e "${BLUE}=== Phase 1: Floating-Point Literals ===${NC}"
echo "=== Floating-Point Literals ===" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# Find floating-point literals (excluding 0.0 and 1.0 which are often intentional)
while IFS=: read -r file line content; do
    # Extract all floating-point numbers from the line
    numbers=$(echo "$content" | grep -oE '\b[0-9]+\.[0-9]+\b' | grep -vE '^(0\.0|1\.0)$')

    for num in $numbers; do
        # Check if this line is a comment or string
        if ! is_comment_or_string "$content" "$num"; then
            # Skip version numbers and certain patterns
            if [[ ! "$content" =~ version ]] && [[ ! "$content" =~ VERSION ]]; then
                category=$(categorize_number "$num")
                echo -e "${YELLOW}$file:$line${NC} - Value: ${RED}$num${NC} (${GREEN}$category${NC})"
                echo "$file:$line - Value: $num ($category)" >> "$OUTPUT_FILE"
                echo "  Context: $content" >> "$OUTPUT_FILE"
                echo "" >> "$OUTPUT_FILE"

                # Clean content for CSV
                clean_content=$(echo "$content" | sed 's/,/ /g' | sed 's/"//g')
                echo "$file,$line,$num,\"$clean_content\",$category" >> "$CSV_FILE"
            fi
        fi
    done
done < <(grep -r -n -E '\b[0-9]+\.[0-9]+\b' src/ --include="*.cpp")

echo ""
echo -e "${BLUE}=== Phase 2: Scientific Notation ===${NC}"
echo "=== Scientific Notation ===" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# Find scientific notation
while IFS=: read -r file line content; do
    numbers=$(echo "$content" | grep -oE '\b[0-9]+(\.[0-9]+)?[eE][+-]?[0-9]+\b')

    for num in $numbers; do
        if ! is_comment_or_string "$content" "$num"; then
            category=$(categorize_number "$num")
            echo -e "${YELLOW}$file:$line${NC} - Value: ${RED}$num${NC} (${GREEN}$category${NC})"
            echo "$file:$line - Value: $num ($category)" >> "$OUTPUT_FILE"
            echo "  Context: $content" >> "$OUTPUT_FILE"
            echo "" >> "$OUTPUT_FILE"

            clean_content=$(echo "$content" | sed 's/,/ /g' | sed 's/"//g')
            echo "$file,$line,$num,\"$clean_content\",$category" >> "$CSV_FILE"
        fi
    done
done < <(grep -r -n -E '\b[0-9]+(\.[0-9]+)?[eE][+-]?[0-9]+\b' src/ --include="*.cpp")

echo ""
echo -e "${BLUE}=== Phase 3: Integer Literals (>10) ===${NC}"
echo "=== Integer Literals (>10) ===" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# Find larger integer literals that might be thresholds or limits
# Process each source file, clean floating-point numbers first, then find integers
find src/ -name "*.cpp" -type f | while read -r filepath; do
    grep -n '.' "$filepath" | while IFS=: read -r line content; do
        # Skip array access patterns like [100] or common loop patterns
        if [[ ! "$content" =~ \[[0-9]+\] ]] && [[ ! "$content" =~ for.*int.*\<.*[0-9]+ ]]; then
            # First, remove all floating-point numbers from the line, then find integers
            # Use more robust regex patterns to handle all floating-point formats including scientific notation
            temp_content=$(echo "$content" | sed -E 's/[0-9]+\.[0-9]+([eE][+-]?[0-9]+)?//g' | sed -E 's/-[0-9]+\.[0-9]+([eE][+-]?[0-9]+)?//g')
            numbers=$(echo "$temp_content" | grep -oE '\b[1-9][0-9]{2,}\b')

            for num in $numbers; do
                if ! is_comment_or_string "$content" "$num"; then
                    category=$(categorize_number "$num")
                    echo -e "${YELLOW}$filepath:$line${NC} - Value: ${RED}$num${NC} (${GREEN}$category${NC})"
                    echo "$filepath:$line - Value: $num ($category)" >> "$OUTPUT_FILE"
                    echo "  Context: $content" >> "$OUTPUT_FILE"
                    echo "" >> "$OUTPUT_FILE"

                    clean_content=$(echo "$content" | sed 's/,/ /g' | sed 's/"//g')
                    echo "$filepath,$line,$num,\"$clean_content\",$category" >> "$CSV_FILE"
                fi
            done
        fi
    done
done

echo ""
echo -e "${BLUE}=== Phase 4: Negative Numbers ===${NC}"
echo "=== Negative Numbers ===" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# Find negative literals (excluding common patterns like n-1)
while IFS=: read -r file line content; do
    # Skip patterns like "n - 1" or "i - 1"
    if [[ ! "$content" =~ [a-zA-Z][[:space:]]*-[[:space:]]*[0-9] ]]; then
        numbers=$(echo "$content" | grep -oE '\-[0-9]+(\.[0-9]+)?\b')

        for num in $numbers; do
            if ! is_comment_or_string "$content" "$num"; then
                category=$(categorize_number "$num")
                echo -e "${YELLOW}$file:$line${NC} - Value: ${RED}$num${NC} (${GREEN}$category${NC})"
                echo "$file:$line - Value: $num ($category)" >> "$OUTPUT_FILE"
                echo "  Context: $content" >> "$OUTPUT_FILE"
                echo "" >> "$OUTPUT_FILE"

                clean_content=$(echo "$content" | sed 's/,/ /g' | sed 's/"//g')
                echo "$file,$line,$num,\"$clean_content\",$category" >> "$CSV_FILE"
            fi
        done
    fi
done < <(grep -r -n -E '\-[0-9]+(\.[0-9]+)?\b' src/ --include="*.cpp")

echo ""
echo "======================================================================="
echo -e "${GREEN}Analysis Complete!${NC}"
echo ""
echo "Summary:"
echo "--------"

# Count by category
echo ""
echo "Magic Numbers by Category:"
for category in mathematical statistical_critical normal_quantile significance_level convergence_tolerance iteration_limit uncategorized; do
    count=$(grep -c ",$category$" "$CSV_FILE" 2>/dev/null)
    if [ $? -eq 0 ] && [ "$count" -gt 0 ]; then
        echo "  $category: $count"
    fi
done

total_count=$(tail -n +2 "$CSV_FILE" | wc -l)
echo ""
echo "Total magic numbers found: $total_count"

echo ""
echo "Reports generated:"
echo "  - Detailed report: $OUTPUT_FILE"
echo "  - CSV for analysis: $CSV_FILE"
echo ""
echo "Next steps:"
echo "  1. Review $CSV_FILE to identify which numbers need constants"
echo "  2. Update constant header files with new definitions"
echo "  3. Run replacement script to update source files"
echo "======================================================================="

#!/usr/bin/env python3
"""
Replace magic numbers with appropriate constants in libstats codebase.

This script identifies common magic numbers and suggests or automatically
replaces them with the appropriate named constants.
"""

import re
import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Mapping of magic numbers to their constant replacements
# Note: We'll be context-aware about when to apply these
FLOAT_MAP = {
    # Basic mathematical constants
    '0.0': 'detail::ZERO_DOUBLE',
    '1.0': 'detail::ONE',
    '2.0': 'detail::TWO',
    '3.0': 'detail::THREE',
    '4.0': 'detail::FOUR',
    '5.0': 'detail::FIVE',
    '6.0': 'detail::SIX',
    '0.5': 'detail::HALF',
    '-0.5': 'detail::NEG_HALF',
    '-1.0': 'detail::NEG_ONE',
    '-2.0': 'detail::NEG_TWO',

    # Chi-squared critical values (α = 0.05)
    '3.841': 'detail::CHI2_95_DF_1',
    '5.991': 'detail::CHI2_95_DF_2',
    '7.815': 'detail::CHI2_95_DF_3',
    '9.488': 'detail::CHI2_95_DF_4',
    '11.070': 'detail::CHI2_95_DF_5',

    # Chi-squared critical values (α = 0.01)
    '6.635': 'detail::CHI2_99_DF_1',
    '9.210': 'detail::CHI2_99_DF_2',

    # Normal quantiles
    '1.645': 'detail::Z_90',
    '1.96': 'detail::Z_95',
    '2.576': 'detail::Z_99',

    # Significance levels
    '0.01': 'detail::ALPHA_01',
    '0.05': 'detail::ALPHA_05',
    '0.10': 'detail::ALPHA_10',
    '0.90': 'detail::CONFIDENCE_90',
    '0.95': 'detail::CONFIDENCE_95',
    '0.99': 'detail::CONFIDENCE_99',
}

INT_MAP = {
    '0': 'detail::ZERO_INT',
    '1': 'detail::ONE_INT',
    '2': 'detail::TWO_INT',
    '3': 'detail::THREE_INT',
    '4': 'detail::FOUR_INT',
    '5': 'detail::FIVE_INT',
    '6': 'detail::SIX_INT',
}

# Patterns to identify magic numbers in different contexts
FLOAT_PATTERN = re.compile(r'\b(-?\d+\.\d+|-?\d+\.0)\b')
INT_PATTERN = re.compile(r'\b([0-6])\b')

# Contexts where we should NOT replace (e.g., array indices, template parameters)
SKIP_CONTEXTS = [
    r'^\s*//',  # Comment lines
    r'^\s*\*',  # Multi-line comment continuation
    r'#include',  # Include statements
    r'constexpr.*=',  # Constant definitions
    r'inline constexpr.*=',  # Inline constexpr definitions
    r'\[\s*\d+\s*\]',  # Array indices
    r'template\s*<',  # Template parameters
    r'case\s+\d+:',  # Switch case labels
    r'\.h:\d+:',  # Line numbers in error messages
    r'//.*',  # Any line with comments (since comments contain contextual numbers)
    r'static const.*=',  # Static const definitions
    r'.*e[+-]\d+',  # Scientific notation - don't replace parts
    r'\b\d+\.\d+e[+-]?\d+\b',  # Full scientific notation patterns
]

def should_skip_line(line: str) -> bool:
    """Check if a line should be skipped based on context."""
    for pattern in SKIP_CONTEXTS:
        if re.search(pattern, line):
            return True
    return False

def is_scientific_notation_context(line: str, match_start: int, match_end: int) -> bool:
    """Check if a number appears to be part of scientific notation."""
    # Check before the match for patterns like "1e" or "1E"
    before = line[:match_start]
    if re.search(r'\de[+-]?$', before, re.IGNORECASE):
        return True

    # Check after the match for patterns like "e5" or "E-5"
    after = line[match_end:]
    if re.search(r'^e[+-]?\d', after, re.IGNORECASE):
        return True

    return False

def is_array_or_template_context(line: str, match_start: int, match_end: int) -> bool:
    """Check if number is used as array index or template parameter."""
    # Check for array index patterns
    before = line[:match_start].rstrip()
    after = line[match_end:].lstrip()

    # Array indices: [...]
    if before.endswith('[') and after.startswith(']'):
        return True

    # Template parameters: <...>
    if '<' in before and '>' in after:
        # Look for template context
        template_depth = 0
        for char in reversed(before):
            if char == '>':
                template_depth += 1
            elif char == '<':
                template_depth -= 1
                if template_depth == 0:
                    return True

    return False

def is_decimal_digit_context(line: str, match_start: int, match_end: int) -> bool:
    """Check if an integer digit is part of a decimal number."""
    # Check if there's a digit followed by a decimal point before our match
    before = line[:match_start]
    if re.search(r'\d+\.$', before):
        return True

    # Check if there's a decimal point followed by digits before our match
    if re.search(r'\.\d*$', before):
        return True

    # Check if there's a decimal point immediately after our match
    after = line[match_end:]
    if after.startswith('.'):
        return True

    return False

def identify_magic_numbers(file_path: Path) -> List[Tuple[int, str, List[str]]]:
    """
    Identify magic numbers in a file.

    Returns list of (line_number, line_content, suggested_replacements)
    """
    results = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading {file_path}: {e}", file=sys.stderr)
        return results

    for i, line in enumerate(lines, 1):
        if should_skip_line(line):
            continue

        suggestions = []

        # Check for floating-point magic numbers with position awareness
        for match in FLOAT_PATTERN.finditer(line):
            number = match.group()
            start, end = match.span()

            # Skip if in scientific notation context
            if is_scientific_notation_context(line, start, end):
                continue

            # Skip if in array/template context
            if is_array_or_template_context(line, start, end):
                continue

            if number in FLOAT_MAP:
                suggestions.append((number, FLOAT_MAP[number]))

        # Check for integer magic numbers with enhanced context checking
        # Be very conservative - only suggest for clear mathematical operations
        if not any(keyword in line.lower() for keyword in ['for', 'while', 'size_t', 'int i', 'vector', 'array']):
            for match in INT_PATTERN.finditer(line):
                number = match.group()
                start, end = match.span()

                # Skip if in array/template context
                if is_array_or_template_context(line, start, end):
                    continue

                # Skip if this digit is part of a decimal number
                if is_decimal_digit_context(line, start, end):
                    continue

                # Only suggest for very clear arithmetic contexts (not all comparisons)
                context_before = line[max(0, start-5):start]
                context_after = line[end:min(len(line), end+5)]

                # Be more restrictive: only arithmetic operations, not all comparisons
                is_arithmetic = (re.search(r'[+\-*/]\s*$', context_before) or
                               re.search(r'^\s*[+\-*/]', context_after))

                # Allow some specific comparison patterns that make sense for constants
                is_meaningful_comparison = (
                    re.search(r'(if|while|return).*==\s*$', context_before) or
                    re.search(r'^\s*(==|!=)', context_after)
                )

                # Skip function calls, constructors, and other contexts
                if '(' in context_before or ')' in context_after:
                    continue

                # Skip cast operations
                if 'static_cast' in line or 'double(' in line or 'int(' in line:
                    continue

                if (is_arithmetic or is_meaningful_comparison) and number in INT_MAP:
                    suggestions.append((number, INT_MAP[number]))

        if suggestions:
            results.append((i, line.rstrip(), suggestions))

    return results

def replace_in_file(file_path: Path, dry_run: bool = True, interactive: bool = False) -> int:
    """
    Replace magic numbers in a file.

    Returns the number of replacements made.
    """
    magic_numbers = identify_magic_numbers(file_path)

    if not magic_numbers:
        return 0

    print(f"\n{'='*60}")
    print(f"File: {file_path}")
    print(f"{'='*60}")

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    replacements_made = 0

    for line_num, original_line, suggestions in magic_numbers:
        print(f"\nLine {line_num}:")
        print(f"  Original: {original_line}")

        modified_line = original_line
        for magic_num, constant in suggestions:
            print(f"  Suggest: {magic_num} -> {constant}")

            if interactive:
                response = input("  Apply this replacement? (y/n/q): ").lower()
                if response == 'q':
                    return replacements_made
                if response != 'y':
                    continue

            # Context-aware replacement - use word boundaries for integers
            if magic_num in INT_MAP:
                # For integers, use word boundaries to avoid replacing parts of decimals
                import re
                pattern = r'\b' + re.escape(magic_num) + r'\b'
                modified_line = re.sub(pattern, constant, modified_line)
            else:
                # For floats, do simple replacement
                modified_line = modified_line.replace(magic_num, constant)
            replacements_made += 1

        if not dry_run and modified_line != original_line:
            lines[line_num - 1] = modified_line + '\n'

    if not dry_run and replacements_made > 0:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        print(f"\nWrote {replacements_made} replacements to {file_path}")

    return replacements_made

def process_directory(directory: Path, pattern: str = "*.cpp",
                     dry_run: bool = True, interactive: bool = False) -> None:
    """Process all matching files in a directory."""
    total_replacements = 0
    files_processed = 0

    for file_path in directory.rglob(pattern):
        if file_path.is_file():
            count = replace_in_file(file_path, dry_run, interactive)
            if count > 0:
                files_processed += 1
                total_replacements += count

    print(f"\n{'='*60}")
    print(f"Summary: {files_processed} files, {total_replacements} potential replacements")
    if dry_run:
        print("(Dry run - no files were modified)")

def main():
    parser = argparse.ArgumentParser(description='Replace magic numbers with constants')
    parser.add_argument('path', help='File or directory to process')
    parser.add_argument('--pattern', default='*.cpp',
                       help='File pattern for directory processing (default: *.cpp)')
    parser.add_argument('--write', action='store_true',
                       help='Actually write changes (default is dry run)')
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Interactive mode - confirm each replacement')

    args = parser.parse_args()

    path = Path(args.path)

    if not path.exists():
        print(f"Error: {path} does not exist", file=sys.stderr)
        sys.exit(1)

    if path.is_file():
        replace_in_file(path, dry_run=not args.write, interactive=args.interactive)
    else:
        process_directory(path, args.pattern,
                         dry_run=not args.write, interactive=args.interactive)

if __name__ == '__main__':
    main()

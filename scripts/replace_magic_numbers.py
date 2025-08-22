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
    '0.0': 'constants::math::ZERO_DOUBLE',
    '1.0': 'constants::math::ONE',
    '2.0': 'constants::math::TWO',
    '3.0': 'constants::math::THREE',
    '4.0': 'constants::math::FOUR',
    '5.0': 'constants::math::FIVE',
    '6.0': 'constants::math::SIX',
    '0.5': 'constants::math::HALF',
    '-0.5': 'constants::math::NEG_HALF',
    '-1.0': 'constants::math::NEG_ONE',
    '-2.0': 'constants::math::NEG_TWO',
}

INT_MAP = {
    '0': 'constants::math::ZERO_INT',
    '1': 'constants::math::ONE_INT',
    '2': 'constants::math::TWO_INT',
    '3': 'constants::math::THREE_INT',
    '4': 'constants::math::FOUR_INT',
    '5': 'constants::math::FIVE_INT',
    '6': 'constants::math::SIX_INT',
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
    r'\[\s*\d+\s*\]',  # Array indices
    r'template\s*<',  # Template parameters
    r'case\s+\d+:',  # Switch case labels
    r'\.h:\d+:',  # Line numbers in error messages
]

def should_skip_line(line: str) -> bool:
    """Check if a line should be skipped based on context."""
    for pattern in SKIP_CONTEXTS:
        if re.search(pattern, line):
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

        # Check for floating-point magic numbers
        float_matches = FLOAT_PATTERN.findall(line)
        for match in float_matches:
            if match in FLOAT_MAP:
                suggestions.append((match, FLOAT_MAP[match]))

        # Check for integer magic numbers in specific contexts
        # (be more conservative with integers to avoid false positives)
        if 'for' not in line and 'i <' not in line and 'i =' not in line:
            # Look for integers used in calculations or comparisons
            if re.search(r'[+\-*/=<>]\s*[0-5]\b', line):
                int_matches = INT_PATTERN.findall(line)
                for match in int_matches:
                    # Skip if it looks like an array index or loop counter
                    if not re.search(rf'\[\s*{match}\s*\]', line):
                        if match in INT_MAP:
                            # Only suggest int constants for integer contexts
                            suggestions.append((match, INT_MAP[match]))

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

            # Simple replacement (could be made smarter with context awareness)
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

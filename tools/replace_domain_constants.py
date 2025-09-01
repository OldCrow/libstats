#!/usr/bin/env python3
"""
Enhanced magic number replacement script for domain-specific statistical constants.
This script identifies and replaces magic numbers with appropriate named constants from
the libstats constant headers.
"""

import re
import sys
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Set

class DomainConstantReplacer:
    def __init__(self):
        # Domain-specific constant mappings
        self.constant_mappings = {
            # Chi-squared critical values
            '3.841': 'detail::CHI2_95_DF_1',
            '5.991': 'detail::CHI2_95_DF_2',
            '7.815': 'detail::CHI2_95_DF_3',
            '9.488': 'detail::CHI2_95_DF_4',
            '11.070': 'detail::CHI2_95_DF_5',

            # Normal distribution critical values
            '1.96': 'detail::Z_95',
            '1.645': 'detail::Z_90',
            '2.576': 'detail::Z_99',
            '2.326': 'detail::Z_99_ONE_TAIL',

            # Statistical significance levels
            '0.01': 'detail::ALPHA_01',
            '0.05': 'detail::ALPHA_05',
            '0.10': 'detail::ALPHA_10',
            '0.001': 'detail::ALPHA_001',

            # Confidence levels
            '0.90': 'detail::CONFIDENCE_90',
            '0.95': 'detail::CONFIDENCE_95',
            '0.99': 'detail::CONFIDENCE_99',
            '0.999': 'detail::CONFIDENCE_999',

            # Common tolerance values
            '1e-6': 'detail::MIN_STD_DEV',
            '1e-8': 'detail::DEFAULT_TOLERANCE',
            '1e-10': 'detail::NEWTON_RAPHSON_TOLERANCE',
            '1e-12': 'detail::HIGH_PRECISION_TOLERANCE',
            '1e-15': 'detail::ULTRA_HIGH_PRECISION_TOLERANCE',
            '1e-300': 'detail::ANDERSON_DARLING_MIN_PROB',
            '1.0e-6': 'detail::MIN_STD_DEV',
            '1.0e-8': 'detail::DEFAULT_TOLERANCE',
            '1.0e-10': 'detail::NEWTON_RAPHSON_TOLERANCE',
            '1.0e-12': 'detail::HIGH_PRECISION_TOLERANCE',
            '1.0e-15': 'detail::ULTRA_HIGH_PRECISION_TOLERANCE',
            '1.0e-300': 'detail::ANDERSON_DARLING_MIN_PROB',

            # Common iteration limits
            '100': 'detail::MAX_NEWTON_ITERATIONS',
            '1000': 'detail::MAX_BISECTION_ITERATIONS',
            '5000': 'detail::MAX_DATA_POINTS_FOR_SW_TEST',

            # Overflow/underflow thresholds
            '700.0': 'detail::LOG_EXP_OVERFLOW_THRESHOLD',
            '37.0': 'detail::LOG1PEXP_LARGE_THRESHOLD',
            '-37.0': 'detail::LOG1PEXP_SMALL_THRESHOLD',

            # Anderson-Darling test constants
            '0.75': 'detail::AD_P_VALUE_MEDIUM',  # When used in AD context
            '0.5': 'detail::AD_THRESHOLD_1',     # When used as AD threshold

            # Kolmogorov-Smirnov approximation
            '0.12': 'detail::KS_APPROX_COEFF_1',
            '0.11': 'detail::KS_APPROX_COEFF_2',

            # Effect size thresholds
            '0.2': 'detail::SMALL_EFFECT',
            '0.8': 'detail::LARGE_EFFECT',

            # Correlation thresholds
            '0.3': 'detail::WEAK_CORRELATION',
            '0.7': 'detail::STRONG_CORRELATION',

            # Power analysis
            '0.80': 'detail::MINIMUM_POWER',
            '0.90': 'detail::HIGH_POWER',

            # Percentile constants
            '0.25': 'detail::QUARTER',
            '100.0': 'detail::HUNDRED',

            # MAD scaling factor (if we have it)
            '1.4826': 'detail::MAD_SCALING_FACTOR',  # Need to check if this exists
        }

        # Context-sensitive mappings (require context analysis)
        self.context_mappings = {
            'chi_squared_critical': {
                '3.841': 'detail::CHI2_95_DF_1',
                '5.991': 'detail::CHI2_95_DF_2',
                '7.815': 'detail::CHI2_95_DF_3',
                '9.488': 'detail::CHI2_95_DF_4',
                '11.070': 'detail::CHI2_95_DF_5',
            },
            'normal_quantile': {
                '1.96': 'detail::Z_95',
                '1.645': 'detail::Z_90',
                '2.576': 'detail::Z_99',
            },
            'significance_level': {
                '0.01': 'detail::ALPHA_01',
                '0.05': 'detail::ALPHA_05',
                '0.10': 'detail::ALPHA_10',
                '0.001': 'detail::ALPHA_001',
            },
            'convergence_tolerance': {
                '1e-6': 'detail::MIN_STD_DEV',
                '1e-8': 'detail::DEFAULT_TOLERANCE',
                '1e-10': 'detail::NEWTON_RAPHSON_TOLERANCE',
                '1e-12': 'detail::HIGH_PRECISION_TOLERANCE',
                '1e-15': 'detail::ULTRA_HIGH_PRECISION_TOLERANCE',
            },
            'iteration_limit': {
                '100': 'detail::MAX_NEWTON_ITERATIONS',
                '1000': 'detail::MAX_BISECTION_ITERATIONS',
                '5000': 'detail::MAX_DATA_POINTS_FOR_SW_TEST',
            }
        }

        # Files requiring specific includes
        self.required_includes = {
            'statistical_constants.h': ['CHI2_', 'Z_', 'T_', 'F_'],
            'threshold_constants.h': ['ALPHA_', 'CONFIDENCE_', 'CORRELATION', 'EFFECT', 'AD_', 'KS_', 'LOG_EXP_OVERFLOW'],
            'precision_constants.h': ['TOLERANCE', 'NEWTON_', 'BISECTION_', 'MAX_.*_ITERATIONS'],
            'mathematical_constants.h': ['QUARTER', 'HUNDRED'],
            'robust_constants.h': ['MAD_SCALING_FACTOR'],
        }

    def load_magic_numbers_csv(self, csv_path: Path) -> List[Dict]:
        """Load magic numbers from CSV file."""
        magic_numbers = []
        try:
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    magic_numbers.append(row)
        except FileNotFoundError:
            print(f"Warning: {csv_path} not found")
        return magic_numbers

    def get_replacement_for_value(self, value: str, category: str = None, context: str = None) -> str:
        """Get the appropriate constant replacement for a magic number."""

        # First try context-specific mapping
        if category and category in self.context_mappings:
            if value in self.context_mappings[category]:
                return self.context_mappings[category][value]

        # Then try general mapping
        if value in self.constant_mappings:
            return self.constant_mappings[value]

        # Special cases based on context
        if context:
            # Anderson-Darling specific
            if 'ad_stat' in context.lower() or 'anderson' in context.lower():
                if value == '0.5':
                    return 'detail::AD_THRESHOLD_1'
                elif value == '0.75':
                    return 'detail::AD_P_VALUE_MEDIUM'

            # Tolerance context
            if 'tolerance' in context.lower() or 'epsilon' in context.lower():
                tolerance_map = {
                    '1e-6': 'detail::MIN_STD_DEV',
                    '1e-8': 'detail::DEFAULT_TOLERANCE',
                    '1e-10': 'detail::NEWTON_RAPHSON_TOLERANCE',
                    '1e-12': 'detail::HIGH_PRECISION_TOLERANCE',
                }
                if value in tolerance_map:
                    return tolerance_map[value]

        return None

    def determine_required_includes(self, replacements: List[str]) -> Set[str]:
        """Determine which header files need to be included."""
        includes = set()

        for replacement in replacements:
            for header, patterns in self.required_includes.items():
                for pattern in patterns:
                    if re.search(pattern, replacement):
                        includes.add(f'#include "../include/core/{header}"')
                        break

        return includes

    def process_file(self, file_path: Path, magic_numbers: List[Dict], write_changes: bool = False) -> Tuple[int, List[str]]:
        """Process a single file and replace magic numbers."""

        try:
            with open(file_path, 'r') as f:
                content = f.read()
                lines = content.splitlines()
        except FileNotFoundError:
            print(f"Warning: {file_path} not found")
            return 0, []

        # Filter magic numbers for this file
        file_str = str(file_path)
        file_magic_numbers = [mn for mn in magic_numbers if mn['File'].replace('//', '/') == file_str or file_str.endswith(mn['File'])]

        if not file_magic_numbers:
            return 0, []

        replacements_made = []
        changes_made = 0

        # Group by line number for efficient processing
        line_replacements = {}
        for mn in file_magic_numbers:
            try:
                line_num = int(mn['Line']) - 1  # Convert to 0-based
                if line_num not in line_replacements:
                    line_replacements[line_num] = []
                line_replacements[line_num].append(mn)
            except (ValueError, KeyError):
                continue

        # Process each line with replacements
        for line_num, magic_nums in line_replacements.items():
            if line_num >= len(lines):
                continue

            line = lines[line_num]
            original_line = line

            # Sort by value length (descending) to avoid partial replacements
            magic_nums.sort(key=lambda x: len(x['Value']), reverse=True)

            for mn in magic_nums:
                value = mn['Value']
                category = mn.get('Category', '')
                context = mn.get('Context', '')

                replacement = self.get_replacement_for_value(value, category, context)

                if replacement:
                    # Create regex pattern that matches the exact numeric value
                    # Handle scientific notation and decimal points
                    escaped_value = re.escape(value)

                    # Look for the value as a standalone number (not part of identifier)
                    pattern = r'\b' + escaped_value + r'\b'

                    if re.search(pattern, line):
                        line = re.sub(pattern, replacement, line)
                        replacements_made.append(replacement)
                        changes_made += 1

            if line != original_line:
                lines[line_num] = line

        # Write changes if requested
        if write_changes and changes_made > 0:
            # Determine required includes
            required_includes = self.determine_required_includes(replacements_made)

            # Add includes if needed (simple approach - add after existing includes)
            if required_includes:
                include_section_end = 0
                for i, line in enumerate(lines):
                    if line.strip().startswith('#include'):
                        include_section_end = i + 1

                # Check if includes are already present
                content_str = '\n'.join(lines)
                for include in list(required_includes):
                    if include in content_str:
                        required_includes.remove(include)

                if required_includes:
                    for include in sorted(required_includes):
                        lines.insert(include_section_end, include)
                        include_section_end += 1

            # Write the modified content
            with open(file_path, 'w') as f:
                f.write('\n'.join(lines))

        return changes_made, replacements_made

def main():
    parser = argparse.ArgumentParser(description='Replace domain-specific magic numbers with named constants')
    parser.add_argument('path', help='File or directory to process')
    parser.add_argument('--csv', default='magic_numbers.csv', help='CSV file with magic numbers')
    parser.add_argument('--write', action='store_true', help='Actually write changes')
    parser.add_argument('--pattern', default='*.cpp', help='File pattern for directory processing')

    args = parser.parse_args()

    replacer = DomainConstantReplacer()

    # Load magic numbers from CSV
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"Error: CSV file {csv_path} not found")
        print("Please run the comprehensive magic number analysis first:")
        print("  ./scripts/find_magic_numbers.sh")
        return 1

    magic_numbers = replacer.load_magic_numbers_csv(csv_path)
    if not magic_numbers:
        print("No magic numbers found in CSV")
        return 0

    # Process files
    path = Path(args.path)
    total_replacements = 0
    files_processed = 0

    if path.is_file():
        files_to_process = [path]
    else:
        files_to_process = list(path.rglob(args.pattern))

    print(f"\n{'='*60}")
    print(f"Domain-Specific Magic Number Replacement")
    print(f"{'='*60}\n")

    for file_path in files_to_process:
        if file_path.is_file() and file_path.suffix == '.cpp':
            replacements, replacement_list = replacer.process_file(file_path, magic_numbers, args.write)

            if replacements > 0:
                files_processed += 1
                total_replacements += replacements
                try:
                    relative_path = file_path.relative_to(Path.cwd())
                except ValueError:
                    relative_path = file_path
                print(f"File: {relative_path}")
                print(f"  Replacements: {replacements}")

                # Show sample replacements
                if replacement_list:
                    sample_replacements = list(set(replacement_list))[:5]  # Show unique ones
                    for replacement in sample_replacements:
                        print(f"    Using: {replacement}")
                print()

    print(f"{'='*60}")
    print(f"Summary: {files_processed} files processed, {total_replacements} replacements made")
    if not args.write:
        print("(Dry run - no files were modified)")
    print(f"{'='*60}")

    return 0

if __name__ == '__main__':
    sys.exit(main())

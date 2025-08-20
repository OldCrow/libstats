#!/usr/bin/env python3
"""
Find files using direct distribution constructors instead of safe factory methods.

This script identifies files in tools/, examples/, and tests/ directories that
are using direct distribution constructors rather than the recommended safe
factory methods (::create() or libstats:: type aliases).

The safe factory methods prevent segfaults from ABI compatibility issues.
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Set, Tuple

# Distribution classes and their safe alternatives
DISTRIBUTIONS = {
    'GaussianDistribution': ['libstats::Gaussian', 'GaussianDistribution::create'],
    'ExponentialDistribution': ['libstats::Exponential', 'ExponentialDistribution::create'],
    'UniformDistribution': ['libstats::Uniform', 'UniformDistribution::create'],
    'PoissonDistribution': ['libstats::Poisson', 'PoissonDistribution::create'],
    'GammaDistribution': ['libstats::Gamma', 'GammaDistribution::create'],
    'DiscreteDistribution': ['libstats::Discrete', 'DiscreteDistribution::create']
}

class ConstructorFinder:
    def __init__(self, libstats_root: str):
        self.root = Path(libstats_root)
        self.results = {}

    def find_direct_constructors(self) -> Dict[str, List[Dict]]:
        """Find all direct constructor usages in target directories."""
        target_dirs = ['tools', 'examples', 'tests']
        results = {dir_name: [] for dir_name in target_dirs}

        for dir_name in target_dirs:
            dir_path = self.root / dir_name
            if not dir_path.exists():
                continue

            cpp_files = list(dir_path.glob('*.cpp')) + list(dir_path.glob('*.h'))

            for file_path in cpp_files:
                issues = self._analyze_file(file_path, dir_name)
                if issues:
                    results[dir_name].extend(issues)

        return results

    def _analyze_file(self, file_path: Path, directory: str) -> List[Dict]:
        """Analyze a single file for direct constructor usage."""
        issues = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"Warning: Could not read {file_path}: {e}")
            return []

        for line_num, line in enumerate(lines, 1):
            line = line.strip()

            # Skip comments and empty lines
            if not line or line.startswith('//') or line.startswith('/*'):
                continue

            # Check for direct constructor calls
            for dist_class, alternatives in DISTRIBUTIONS.items():
                # Check if this line contains the distribution class name
                if dist_class in line:
                    # Skip if already using safe methods
                    if '::create(' in line:
                        continue  # Already using safe factory
                    if 'libstats::' in line:
                        continue  # Already using namespace alias

                    # Match various constructor patterns
                    patterns = [
                        # Direct constructor call: DistributionName(args)
                        rf'\b{dist_class}\s*\([^)]*\)',
                        # Variable declaration: DistributionName variable_name(args)
                        rf'\b{dist_class}\s+\w+\s*\([^)]*\)',
                        # Auto assignment: auto var = DistributionName(args)
                        rf'auto\s+\w+\s*=\s*{dist_class}\s*\([^)]*\)',
                        # Member initialization: member_(DistributionName(args))
                        rf'\w+_\({dist_class}\s*\([^)]*\)\)',
                    ]

                    for pattern in patterns:
                        if re.search(pattern, line):
                            issues.append({
                                'file': str(file_path.relative_to(self.root)),
                                'directory': directory,
                                'line_number': line_num,
                                'line_content': line.strip(),
                                'distribution': dist_class,
                                'alternatives': alternatives,
                                'pattern_matched': pattern
                            })
                            break  # Found one pattern, don't match others

        return issues

    def find_safe_usage_examples(self) -> Dict[str, List[str]]:
        """Find examples of correct safe factory usage."""
        examples = {}

        # Check examples directory for good patterns
        examples_dir = self.root / 'examples'
        if examples_dir.exists():
            for file_path in examples_dir.glob('*.cpp'):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    for dist_class, alternatives in DISTRIBUTIONS.items():
                        for alt in alternatives:
                            if alt in content:
                                if dist_class not in examples:
                                    examples[dist_class] = []
                                examples[dist_class].append(f"{file_path.name}: {alt}")

                except Exception:
                    continue

        return examples

def generate_conversion_report(results: Dict, safe_examples: Dict, output_file: str):
    """Generate a detailed conversion report."""

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Distribution Constructor Safety Conversion Report\n\n")
        f.write("This report identifies files using direct distribution constructors that should be converted to safe factory methods.\n\n")

        total_issues = sum(len(issues) for issues in results.values())
        f.write(f"## Summary\n\n")
        f.write(f"**Total issues found:** {total_issues}\n\n")

        for directory, issues in results.items():
            if issues:
                f.write(f"- **{directory}/:** {len(issues)} issues\n")

        f.write("\n## Safe Factory Method Examples\n\n")
        f.write("The following safe patterns should be used instead of direct constructors:\n\n")

        for dist_class, alternatives in DISTRIBUTIONS.items():
            f.write(f"### {dist_class}\n\n")
            f.write("**Safe alternatives:**\n")
            for alt in alternatives:
                if '::create' in alt:
                    f.write(f"- `{alt}(params)` - Exception-free factory method\n")
                else:
                    f.write(f"- `{alt}(params)` - Type alias (uses safe construction internally)\n")
            f.write("\n")

            if dist_class in safe_examples:
                f.write("**Examples of correct usage found:**\n")
                for example in safe_examples[dist_class][:3]:  # Show up to 3 examples
                    f.write(f"- {example}\n")
                f.write("\n")

        f.write("## Files Requiring Conversion\n\n")

        for directory, issues in results.items():
            if not issues:
                continue

            f.write(f"### {directory}/ Directory\n\n")

            # Group by file
            files_grouped = {}
            for issue in issues:
                file_name = issue['file']
                if file_name not in files_grouped:
                    files_grouped[file_name] = []
                files_grouped[file_name].append(issue)

            for file_name, file_issues in files_grouped.items():
                f.write(f"#### `{file_name}`\n\n")
                f.write(f"**Issues found:** {len(file_issues)}\n\n")

                for issue in file_issues:
                    f.write(f"**Line {issue['line_number']}:** `{issue['distribution']}`\n")
                    f.write(f"```cpp\n{issue['line_content']}\n```\n")
                    f.write("**Recommended alternatives:**\n")
                    for alt in issue['alternatives']:
                        f.write(f"- `{alt}(...)`\n")
                    f.write("\n")

def main():
    import sys

    # Get libstats root directory
    if len(sys.argv) > 1:
        libstats_root = sys.argv[1]
    else:
        # Assume we're in the tools directory
        libstats_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    print(f"Analyzing libstats directory: {libstats_root}")

    finder = ConstructorFinder(libstats_root)

    print("Scanning for direct constructor usage...")
    results = finder.find_direct_constructors()

    print("Finding safe usage examples...")
    safe_examples = finder.find_safe_usage_examples()

    # Print summary to console
    total_issues = sum(len(issues) for issues in results.values())
    print(f"\n=== ANALYSIS COMPLETE ===")
    print(f"Total issues found: {total_issues}")

    for directory, issues in results.items():
        if issues:
            print(f"  {directory}/: {len(issues)} issues")

    # Generate detailed report
    output_file = os.path.join(libstats_root, 'docs', 'constructor_safety_conversion_report.md')
    generate_conversion_report(results, safe_examples, output_file)
    print(f"\nDetailed report generated: {output_file}")

    # Show some examples
    if total_issues > 0:
        print("\n=== SAMPLE ISSUES (first 5) ===")
        count = 0
        for directory, issues in results.items():
            for issue in issues:
                if count >= 5:
                    break
                print(f"{issue['file']}:{issue['line_number']} - {issue['distribution']}")
                print(f"  Current: {issue['line_content']}")
                print(f"  Use instead: {issue['alternatives'][0]}")
                print()
                count += 1
            if count >= 5:
                break

if __name__ == '__main__':
    main()

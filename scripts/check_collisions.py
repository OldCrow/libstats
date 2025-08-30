#!/usr/bin/env python3
"""Check for potential naming collisions when collapsing namespaces."""

import os
import re
from collections import defaultdict
from pathlib import Path

def extract_symbols(file_path):
    """Extract function, class, struct, enum, and variable declarations."""
    symbols = defaultdict(list)

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # Remove comments
    content = re.sub(r'//.*?\n', '\n', content)
    content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)

    # Extract class/struct names
    for match in re.finditer(r'\b(class|struct)\s+([A-Z]\w+)\b', content):
        symbol_type = match.group(1)
        symbol_name = match.group(2)
        symbols['types'].append({
            'name': symbol_name,
            'kind': symbol_type,
            'file': str(file_path)
        })

    # Extract enum names
    for match in re.finditer(r'\benum\s+(?:class\s+)?([A-Z]\w+)\b', content):
        symbols['types'].append({
            'name': match.group(1),
            'kind': 'enum',
            'file': str(file_path)
        })

    # Extract function declarations (simplified)
    for match in re.finditer(r'\b(\w+)\s+(\w+)\s*\([^)]*\)\s*(?:const\s*)?(?:noexcept\s*)?[;{]', content):
        return_type = match.group(1)
        func_name = match.group(2)
        # Skip constructors/destructors and common keywords
        if (func_name[0].islower() and
            return_type not in ['if', 'for', 'while', 'switch', 'return', 'namespace', 'using']):
            symbols['functions'].append({
                'name': func_name,
                'file': str(file_path)
            })

    # Extract global constants (simplified)
    for match in re.finditer(r'(?:inline\s+)?(?:constexpr|const)\s+\w+\s+([A-Z_][A-Z0-9_]*)\s*=', content):
        symbols['constants'].append({
            'name': match.group(1),
            'file': str(file_path)
        })

    return symbols

def analyze_collisions(root_dir):
    """Analyze potential naming collisions."""
    all_symbols = defaultdict(lambda: defaultdict(list))

    for root, dirs, files in os.walk(root_dir):
        # Skip build directories
        if 'build' in root or '.git' in root:
            continue

        for file in files:
            if file.endswith(('.h', '.cpp', '.hpp')):
                file_path = Path(root) / file
                try:
                    symbols = extract_symbols(file_path)
                    for category, items in symbols.items():
                        for item in items:
                            all_symbols[category][item['name']].append(item)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

    return all_symbols

def print_collision_report(all_symbols):
    """Print collision analysis report."""
    print("=" * 80)
    print("NAMING COLLISION ANALYSIS")
    print("=" * 80)

    total_collisions = 0

    for category in ['types', 'functions', 'constants']:
        collisions = {name: occurrences for name, occurrences in all_symbols[category].items()
                     if len(occurrences) > 1}

        if collisions:
            print(f"\n{category.upper()} with multiple definitions:")
            print("-" * 40)

            # Sort by number of occurrences
            sorted_collisions = sorted(collisions.items(),
                                      key=lambda x: len(x[1]), reverse=True)

            for name, occurrences in sorted_collisions[:20]:
                print(f"  {name}: {len(occurrences)} occurrences")
                # Show files for first few
                if len(sorted_collisions) <= 10:
                    unique_files = set()
                    for occ in occurrences:
                        file_path = occ['file']
                        # Simplify path
                        if './include/' in file_path:
                            file_path = file_path.split('./include/')[-1]
                        elif './src/' in file_path:
                            file_path = file_path.split('./src/')[-1]
                        unique_files.add(file_path)
                    for f in sorted(unique_files)[:3]:
                        print(f"    - {f}")
                    if len(unique_files) > 3:
                        print(f"    ... and {len(unique_files) - 3} more files")

            if len(sorted_collisions) > 20:
                print(f"  ... and {len(sorted_collisions) - 20} more")

            total_collisions += len(collisions)

    print("\n" + "=" * 80)
    print(f"SUMMARY: {total_collisions} potential naming collisions found")
    print("=" * 80)

    # Recommendations
    print("\nRECOMMENDATIONS:")
    print("-" * 40)
    if total_collisions < 50:
        print("✓ Low collision count - namespace flattening is feasible")
        print("  Most collisions appear to be intentional (same interface in different contexts)")
    elif total_collisions < 100:
        print("⚠ Moderate collision count - namespace flattening requires some refactoring")
        print("  Consider prefixing some symbols to disambiguate")
    else:
        print("✗ High collision count - significant refactoring needed")
        print("  Consider keeping some sub-namespaces for logical grouping")

if __name__ == "__main__":
    root_directory = "."
    symbols = analyze_collisions(root_directory)
    print_collision_report(symbols)

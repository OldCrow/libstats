#!/usr/bin/env python3
"""
Analyze magic number distribution across the codebase.
"""

import subprocess
import re
from pathlib import Path

def analyze_magic_numbers():
    """Run magic number analysis and parse results."""
    result = subprocess.run([
        'python3', 'tools/replace_magic_numbers.py', 'src/', '--pattern', '*.cpp'
    ], capture_output=True, text=True, cwd=Path.cwd())

    if result.returncode != 0:
        print("Error running magic number script:", result.stderr)
        return

    output = result.stdout
    files = {}
    current_file = None

    for line in output.split('\n'):
        if line.startswith('File: '):
            current_file = line.replace('File: ', '').strip()
            files[current_file] = 0
        elif line.startswith('Line ') and current_file:
            files[current_file] += 1

    # Sort by number of suggestions (descending)
    sorted_files = sorted(files.items(), key=lambda x: x[1], reverse=True)

    print("Magic Number Distribution by File:")
    print("=" * 50)
    total_suggestions = 0

    for file_path, count in sorted_files:
        if count > 0:
            print(f"{file_path}: {count} suggestions")
            total_suggestions += count

    print("=" * 50)
    print(f"Total files with magic numbers: {len([f for f in files.values() if f > 0])}")
    print(f"Total magic number suggestions: {total_suggestions}")

    # Categorize files by suggestion count
    high = [f for f, c in sorted_files if c >= 50]
    medium = [f for f, c in sorted_files if 20 <= c < 50]
    low = [f for f, c in sorted_files if 5 <= c < 20]
    minimal = [f for f, c in sorted_files if 1 <= c < 5]

    print(f"\nFile Categories:")
    print(f"High (50+ suggestions): {len(high)} files")
    print(f"Medium (20-49 suggestions): {len(medium)} files")
    print(f"Low (5-19 suggestions): {len(low)} files")
    print(f"Minimal (1-4 suggestions): {len(minimal)} files")

    if high:
        print(f"\nHigh-impact files:")
        for f in high:
            count = next(c for file, c in sorted_files if file == f)
            print(f"  {f}: {count}")

if __name__ == '__main__':
    analyze_magic_numbers()

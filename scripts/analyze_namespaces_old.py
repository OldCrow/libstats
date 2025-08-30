#!/usr/bin/env python3
"""Analyze namespace structure in libstats C++ codebase."""

import os
import re
from collections import defaultdict
from pathlib import Path

def extract_namespaces(file_path):
    """Extract namespace information from a file."""
    namespaces = []
    namespace_stack = []

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    for line_num, line in enumerate(lines, 1):
        # Skip comments
        if line.strip().startswith('//'):
            continue

        # Match namespace declarations
        ns_match = re.match(r'\s*namespace\s+(\w+)\s*\{', line)
        if ns_match:
            ns_name = ns_match.group(1)
            namespace_stack.append(ns_name)
            full_ns = '::'.join(namespace_stack)
            namespaces.append({
                'file': str(file_path),
                'line': line_num,
                'namespace': full_ns,
                'level': len(namespace_stack)
            })

        # Match closing braces (simplified - may not be perfect)
        if re.search(r'^\s*\}\s*//\s*namespace', line):
            if namespace_stack:
                namespace_stack.pop()

    return namespaces

def analyze_codebase(root_dir):
    """Analyze all C++ files in the codebase."""
    all_namespaces = []
    namespace_contents = defaultdict(lambda: {'files': set(), 'count': 0})

    for root, dirs, files in os.walk(root_dir):
        # Skip build directories
        if 'build' in root or '.git' in root:
            continue

        for file in files:
            if file.endswith(('.h', '.cpp', '.hpp')):
                file_path = Path(root) / file
                try:
                    ns_list = extract_namespaces(file_path)
                    for ns_info in ns_list:
                        all_namespaces.append(ns_info)
                        ns = ns_info['namespace']
                        namespace_contents[ns]['files'].add(ns_info['file'])
                        namespace_contents[ns]['count'] += 1
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

    return all_namespaces, namespace_contents

def print_analysis(all_namespaces, namespace_contents):
    """Print analysis results."""
    print("=" * 80)
    print("NAMESPACE ANALYSIS REPORT")
    print("=" * 80)

    # Count unique namespaces
    unique_namespaces = set(ns['namespace'] for ns in all_namespaces)
    print(f"\nTotal unique namespaces: {len(unique_namespaces)}")

    # Analyze depth
    max_depth = max((ns['level'] for ns in all_namespaces), default=0)
    print(f"Maximum namespace depth: {max_depth}")

    # Group by top-level namespace
    top_level = defaultdict(set)
    for ns in unique_namespaces:
        parts = ns.split('::')
        top_level[parts[0]].add(ns)

    print(f"\nTop-level namespaces: {len(top_level)}")
    print("-" * 40)
    for tl in sorted(top_level.keys()):
        sub_count = len(top_level[tl])
        print(f"  {tl}: {sub_count} namespace(s)")
        if sub_count > 1 and sub_count <= 10:
            for sub_ns in sorted(top_level[tl]):
                if sub_ns != tl:
                    print(f"    - {sub_ns}")
        elif sub_count > 10:
            # Show first few
            shown = 0
            for sub_ns in sorted(top_level[tl]):
                if sub_ns != tl and shown < 5:
                    print(f"    - {sub_ns}")
                    shown += 1
            print(f"    ... and {sub_count - 5} more")

    # Find most commonly used namespaces
    print("\n" + "=" * 40)
    print("Most frequently used namespaces:")
    print("-" * 40)
    sorted_ns = sorted(namespace_contents.items(), key=lambda x: x[1]['count'], reverse=True)
    for ns, info in sorted_ns[:20]:
        print(f"  {ns}: used {info['count']} times in {len(info['files'])} file(s)")

    # Identify deep hierarchies
    deep_namespaces = [ns for ns in unique_namespaces if ns.count('::') >= 2]
    if deep_namespaces:
        print("\n" + "=" * 40)
        print(f"Deep namespace hierarchies (3+ levels): {len(deep_namespaces)}")
        print("-" * 40)
        for ns in sorted(deep_namespaces)[:10]:
            print(f"  {ns}")
        if len(deep_namespaces) > 10:
            print(f"  ... and {len(deep_namespaces) - 10} more")

if __name__ == "__main__":
    root_directory = "."
    all_ns, ns_contents = analyze_codebase(root_directory)
    print_analysis(all_ns, ns_contents)

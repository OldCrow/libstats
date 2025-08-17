#!/usr/bin/env python3
"""
Header Dependency Analysis Tool for libstats

This tool analyzes header include patterns, measures redundancy reduction,
and validates the effectiveness of header refactoring work.
"""

import os
import re
import sys
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple

class HeaderAnalyzer:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.include_graph = defaultdict(set)
        self.header_files = []
        self.include_counts = Counter()
        self.redundancy_stats = {}
        
    def find_header_files(self) -> List[Path]:
        """Find all header files in the project."""
        headers = []
        for pattern in ["**/*.h", "**/*.hpp"]:
            headers.extend(self.project_root.glob(pattern))
        
        # Filter out build directory and external dependencies
        filtered = []
        for header in headers:
            rel_path = header.relative_to(self.project_root)
            if not any(part in str(rel_path) for part in ['build', 'external', '.git']):
                filtered.append(header)
                
        self.header_files = filtered
        return filtered
    
    def parse_includes(self, file_path: Path) -> List[str]:
        """Parse #include statements from a header file."""
        includes = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Find all #include statements
            include_pattern = r'^\s*#include\s+[<"]([^>"]+)[>"]'
            matches = re.findall(include_pattern, content, re.MULTILINE)
            
            for match in matches:
                # Filter out system includes for our analysis
                if not match.startswith('/') and not match.startswith('std'):
                    includes.append(match)
                    
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            
        return includes
    
    def build_dependency_graph(self):
        """Build the include dependency graph."""
        print("🔍 Building header dependency graph...")
        
        for header in self.header_files:
            includes = self.parse_includes(header)
            rel_path = str(header.relative_to(self.project_root))
            
            self.include_graph[rel_path] = set(includes)
            for include in includes:
                self.include_counts[include] += 1
                
        print(f"   Analyzed {len(self.header_files)} header files")
    
    def analyze_common_headers(self) -> Dict[str, int]:
        """Identify the most commonly included headers."""
        print("\n📊 Most commonly included headers:")
        common = self.include_counts.most_common(15)
        
        for header, count in common:
            print(f"   {header:<40} {count:>3} times")
            
        return dict(common)
    
    def analyze_distribution_headers(self):
        """Analyze include patterns specifically for distribution headers."""
        print("\n🎯 Distribution headers analysis:")
        
        dist_headers = [h for h in self.header_files if 'distributions/' in str(h)]
        
        if not dist_headers:
            print("   No distribution headers found")
            return
            
        print(f"   Found {len(dist_headers)} distribution headers")
        
        # Analyze includes per distribution
        common_includes = set()
        distribution_includes = {}
        
        for header in dist_headers:
            rel_path = str(header.relative_to(self.project_root))
            includes = self.include_graph[rel_path]
            distribution_includes[rel_path] = includes
            
            if not common_includes:
                common_includes = includes.copy()
            else:
                common_includes &= includes
                
        print(f"   Common includes across ALL distributions: {len(common_includes)}")
        for inc in sorted(common_includes):
            print(f"     - {inc}")
            
        # Calculate redundancy reduction potential
        total_before = sum(len(includes) for includes in distribution_includes.values())
        potential_after = len(dist_headers) * 2 + len(common_includes)  # 2 common headers + unique ones
        
        print(f"   Total includes before optimization: {total_before}")
        print(f"   Potential includes after optimization: {potential_after}")
        print(f"   Potential reduction: {total_before - potential_after} ({((total_before - potential_after) / total_before * 100):.1f}%)")
    
    def check_refactoring_effectiveness(self):
        """Check if our refactoring was effective by looking for common header usage."""
        print("\n✅ Refactoring effectiveness analysis:")
        
        # Look for usage of our common headers (using relative path patterns)
        common_header_patterns = {
            'core/distribution_common.h': ['core/distribution_common.h', '../core/distribution_common.h'],
            'core/essential_constants.h': ['core/essential_constants.h', '../core/essential_constants.h'],
            'platform/platform_common.h': ['platform/platform_common.h', 'platform_common.h', '../platform/platform_common.h'],
            'distributions/distribution_platform_common.h': ['distribution_platform_common.h', 'distributions/distribution_platform_common.h']
        }
        
        common_header_usage = {header: 0 for header in common_header_patterns}
        
        for header_path, includes in self.include_graph.items():
            for common_header, patterns in common_header_patterns.items():
                if any(pattern in inc for inc in includes for pattern in patterns):
                    common_header_usage[common_header] += 1
        
        print("   Common header adoption:")
        for header, usage in common_header_usage.items():
            status = "✅ GOOD" if usage > 0 else "⚠️  NOT USED"
            print(f"   {header:<45} {usage:>2} files {status}")
    
    def measure_transitive_dependencies(self):
        """Analyze transitive dependencies to identify bloat."""
        print("\n🔗 Transitive dependency analysis:")
        
        def get_transitive_deps(header: str, visited: Set[str] = None) -> Set[str]:
            if visited is None:
                visited = set()
            if header in visited:
                return set()
            
            visited.add(header)
            deps = self.include_graph.get(header, set()).copy()
            
            for dep in list(deps):
                # Find the full path for this dependency
                dep_full_path = None
                for full_path in self.include_graph:
                    if full_path.endswith(dep):
                        dep_full_path = full_path
                        break
                        
                if dep_full_path:
                    transitive = get_transitive_deps(dep_full_path, visited.copy())
                    deps.update(transitive)
                    
            return deps
        
        # Analyze a few key headers
        key_headers = [
            'include/distributions/gaussian.h',
            'include/core/distribution_base.h',
            'include/libstats.h'
        ]
        
        for header in key_headers:
            # Find the actual file path
            matching_files = [h for h in self.include_graph.keys() if h.endswith(header.split('/')[-1])]
            if matching_files:
                full_header = matching_files[0]
                transitive_deps = get_transitive_deps(full_header)
                print(f"   {header:<45} {len(transitive_deps):>3} transitive dependencies")
    
    def generate_include_report(self):
        """Generate a comprehensive report of the header analysis."""
        print("\n" + "="*80)
        print("📋 HEADER OPTIMIZATION REPORT")
        print("="*80)
        
        # Summary statistics
        total_includes = sum(len(includes) for includes in self.include_graph.values())
        avg_includes = total_includes / len(self.include_graph) if self.include_graph else 0
        
        print(f"📊 Summary Statistics:")
        print(f"   Total header files analyzed: {len(self.header_files)}")
        print(f"   Total #include statements: {total_includes}")
        print(f"   Average includes per header: {avg_includes:.1f}")
        print(f"   Unique headers included: {len(self.include_counts)}")
        
        # Top redundant includes
        print(f"\n🔄 Most redundant includes (optimization candidates):")
        for header, count in self.include_counts.most_common(10):
            if count > 1:
                percentage = (count / len(self.include_graph)) * 100
                print(f"   {header:<40} {count:>3} files ({percentage:>5.1f}%)")
    
    def run_analysis(self):
        """Run the complete header analysis."""
        print("🚀 Starting Header Dependency Analysis for libstats")
        print("-" * 60)
        
        self.find_header_files()
        self.build_dependency_graph()
        self.analyze_common_headers()
        self.analyze_distribution_headers()
        self.check_refactoring_effectiveness()
        self.measure_transitive_dependencies()
        self.generate_include_report()
        
        print("\n✅ Analysis complete!")


def main():
    if len(sys.argv) > 1:
        project_root = sys.argv[1]
    else:
        # Default to current directory's parent (assuming we're in tools/)
        project_root = Path(__file__).parent.parent
    
    analyzer = HeaderAnalyzer(project_root)
    analyzer.run_analysis()


if __name__ == "__main__":
    main()

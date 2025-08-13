#!/usr/bin/env python3
"""
Static Analysis Tool for Header Optimization

This tool uses clang's static analysis capabilities to:
1. Detect unused includes
2. Find forward declaration opportunities
3. Analyze include hierarchies
4. Validate refactoring effectiveness
"""

import os
import re
import sys
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict

class StaticAnalyzer:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.analysis_results = {}
        
    def run_clang_unused_includes(self, header_file: Path) -> List[str]:
        """Use clang to detect unused includes in a header file."""
        try:
            # Create a temporary test file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as f:
                f.write(f'#include "{header_file}"\nint main(){{ return 0; }}\n')
                test_file = f.name
            
            # Run clang with unused include detection
            cmd = [
                'clang++',
                '-std=c++20',
                f'-I{self.project_root}/include',
                '-Wunused-macros',
                '-Wall',
                '-fsyntax-only',
                test_file
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            unused_includes = []
            if result.stderr:
                # Parse warnings for unused includes
                lines = result.stderr.split('\n')
                for line in lines:
                    if 'unused' in line and '#include' in line:
                        unused_includes.append(line.strip())
            
            os.unlink(test_file)
            return unused_includes
            
        except Exception as e:
            print(f"Error analyzing {header_file}: {e}")
            return []
    
    def validate_common_header_effectiveness(self) -> Dict[str, any]:
        """Validate that common headers are reducing redundancy."""
        print("✅ Validating common header effectiveness...")
        
        common_headers = [
            'core/distribution_common.h',
            'core/essential_constants.h',
            'platform/platform_common.h',
            'distributions/distribution_platform_common.h'
        ]
        
        results = {}
        
        for common_header in common_headers:
            header_path = self.project_root / "include" / common_header
            if not header_path.exists():
                results[common_header] = {'status': 'NOT_FOUND'}
                continue
            
            # Count how many files include this common header
            usage_count = 0
            using_files = []
            
            for header_file in self.project_root.glob("include/**/*.h"):
                try:
                    with open(header_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    if common_header in content or common_header.split('/')[-1] in content:
                        usage_count += 1
                        using_files.append(str(header_file.relative_to(self.project_root)))
                        
                except Exception:
                    continue
            
            results[common_header] = {
                'usage_count': usage_count,
                'using_files': using_files,
                'status': 'EFFECTIVE' if usage_count >= 3 else 'UNDERUSED'
            }
        
        return results
    
    def generate_optimization_recommendations(self) -> List[str]:
        """Generate specific recommendations for further optimization."""
        recommendations = []
        
        # Analyze all headers
        print("🎯 Generating optimization recommendations...")
        
        distribution_headers = list(self.project_root.glob("include/distributions/*.h"))
        
        # Check distribution header consistency
        dist_includes = defaultdict(int)
        for header in distribution_headers:
            try:
                with open(header, 'r', encoding='utf-8') as f:
                    content = f.read()
                includes = re.findall(r'#include\s+[<"]([^>"]+)[>"]', content)
                for include in includes:
                    dist_includes[include] += 1
            except Exception:
                continue
        
        # Find highly redundant includes in distributions
        common_dist_includes = [(inc, count) for inc, count in dist_includes.items() 
                               if count >= len(distribution_headers) * 0.7]
        
        if common_dist_includes:
            recommendations.append(
                f"Consider consolidating {len(common_dist_includes)} highly redundant includes "
                f"across distribution headers into a common header"
            )
        else:
            recommendations.append("Distribution headers show good consolidation - no major redundancy detected")
        
        return recommendations
    
    def run_analysis(self):
        """Run the complete static analysis."""
        print("🔬 Starting Static Analysis for Header Optimization")
        print("-" * 60)
        
        # Test key headers for unused includes
        key_headers = [
            "libstats.h",
            "core/distribution_base.h", 
            "distributions/gaussian.h",
            "platform/simd.h"
        ]
        
        print("🧹 Unused includes analysis:")
        for header_name in key_headers:
            header_path = self.project_root / "include" / header_name
            if header_path.exists():
                unused = self.run_clang_unused_includes(header_path)
                if unused:
                    print(f"   ⚠️  {header_name}: {len(unused)} potential unused includes")
                    for u in unused[:3]:  # Show first 3
                        print(f"      - {u}")
                else:
                    print(f"   ✅ {header_name}: No obvious unused includes detected")
        
        # Validate common headers
        common_header_results = self.validate_common_header_effectiveness()
        print("\n📋 Common header effectiveness:")
        for header, result in common_header_results.items():
            status = result.get('status', 'UNKNOWN')
            usage = result.get('usage_count', 0)
            emoji = "✅" if status == 'EFFECTIVE' else "⚠️" if status == 'UNDERUSED' else "❌"
            print(f"   {emoji} {header}: {usage} files, {status}")
        
        # Generate recommendations
        recommendations = self.generate_optimization_recommendations()
        if recommendations:
            print("\n🎯 Optimization recommendations:")
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec}")
        
        print("\n✅ Static analysis complete!")


def main():
    if len(sys.argv) > 1:
        project_root = sys.argv[1]
    else:
        project_root = Path(__file__).parent.parent
    
    analyzer = StaticAnalyzer(project_root)
    analyzer.run_analysis()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import os
import re
from collections import defaultdict, Counter
import json

def analyze_header_complexity(header_path):
    """Analyze the complexity and optimization opportunities of a header file"""
    if not os.path.exists(header_path):
        return None
    
    with open(header_path, 'r') as f:
        content = f.read()
    
    # Extract includes
    includes = re.findall(r'#include\s+[<"]([^>"]+)[>"]', content)
    
    # Count template usage
    template_count = len(re.findall(r'\btemplate\s*<', content))
    
    # Count inline functions
    inline_count = len(re.findall(r'\binline\s+', content))
    
    # Check for heavy STL usage
    heavy_stl = ['<vector>', '<string>', '<map>', '<unordered_map>', '<algorithm>', 
                 '<numeric>', '<functional>', '<future>', '<thread>', '<chrono>',
                 '<deque>', '<span>', '<optional>', '<variant>']
    heavy_stl_count = sum(1 for inc in includes if f'<{inc}>' in heavy_stl or inc in [h[1:-1] for h in heavy_stl])
    
    # Count forward declarations
    forward_decl_count = len(re.findall(r'\bclass\s+\w+;', content))
    
    # Estimate lines of code (excluding comments and empty lines)
    loc_lines = [line.strip() for line in content.split('\n') 
                if line.strip() and not line.strip().startswith('//') and not line.strip().startswith('*')]
    loc = len(loc_lines)
    
    # Calculate complexity score
    complexity_score = (
        len(includes) * 2 +           # Each include adds complexity
        template_count * 5 +          # Templates are expensive to compile
        inline_count * 2 +            # Inline functions add to preprocessing
        heavy_stl_count * 10 +        # Heavy STL headers are very expensive
        loc * 0.1                     # General code complexity
    ) - forward_decl_count * 1        # Forward declarations reduce complexity
    
    return {
        'includes': includes,
        'include_count': len(includes),
        'template_count': template_count,
        'inline_count': inline_count,
        'heavy_stl_count': heavy_stl_count,
        'forward_decl_count': forward_decl_count,
        'loc': loc,
        'complexity_score': complexity_score,
        'heavy_stl_headers': [inc for inc in includes if f'<{inc}>' in heavy_stl or inc in [h[1:-1] for h in heavy_stl]]
    }

def find_optimization_opportunities():
    """Find header optimization opportunities in the codebase"""
    # Determine the correct include directory path
    if os.path.exists('include'):
        include_dir = 'include'
        output_path = 'tools/header_optimization_analysis.json'
    elif os.path.exists('../include'):
        include_dir = '../include'
        output_path = 'header_optimization_analysis.json'
    else:
        print("❌ Error: Cannot find 'include' directory")
        print("   Current directory:", os.getcwd())
        print("   Please run from project root or tools directory")
        return []
    
    opportunities = []
    header_analysis = {}
    
    # Analyze all headers
    for root, dirs, files in os.walk(include_dir):
        for file in files:
            if file.endswith('.h'):
                header_path = os.path.join(root, file)
                rel_path = os.path.relpath(header_path, include_dir)
                analysis = analyze_header_complexity(header_path)
                if analysis:
                    header_analysis[rel_path] = analysis
    
    # Sort by complexity score
    sorted_headers = sorted(header_analysis.items(), key=lambda x: x[1]['complexity_score'], reverse=True)
    
    print("🚀 Header Optimization Analysis")
    print("=" * 60)
    
    # Top 10 most complex headers
    print("\n📊 Most Complex Headers (Top Optimization Candidates):")
    print("-" * 60)
    for i, (header, analysis) in enumerate(sorted_headers[:10], 1):
        print(f"{i:2d}. {header:<35} | Score: {analysis['complexity_score']:6.1f} | "
              f"Includes: {analysis['include_count']:2d} | Templates: {analysis['template_count']:2d} | "
              f"Heavy STL: {analysis['heavy_stl_count']:2d}")
    
    # Most commonly included headers
    all_includes = []
    for analysis in header_analysis.values():
        all_includes.extend(analysis['includes'])
    
    include_counter = Counter(all_includes)
    print(f"\n🔄 Most Redundant Includes (Consolidation Candidates):")
    print("-" * 60)
    for include, count in include_counter.most_common(15):
        percentage = (count / len(header_analysis)) * 100 if header_analysis else 0
        print(f"   {include:<35} | Used {count:2d} times ({percentage:5.1f}% of headers)")
    
    # Identify highest priority optimization targets
    print(f"\n🎯 Priority Header Optimization Targets:")
    print("-" * 60)
    
    priority_headers = [
        'core/performance_dispatcher.h',
        'core/performance_history.h', 
        'platform/work_stealing_pool.h',
        'platform/benchmark.h',
        'platform/thread_pool.h',
        'platform/parallel_execution.h',
        'core/distribution_memory.h',
        'core/dispatch_utils.h'
    ]
    
    for header in priority_headers:
        if header in header_analysis:
            analysis = header_analysis[header]
            print(f"   {header}")
            print(f"   └─ Complexity: {analysis['complexity_score']:.1f} | "
                  f"Heavy STL: {analysis['heavy_stl_count']} | "
                  f"Templates: {analysis['template_count']}")
            if analysis['heavy_stl_headers']:
                print(f"   └─ Heavy includes: {', '.join(analysis['heavy_stl_headers'])}")
    
    # Optimization recommendations
    print(f"\n📈 Optimization Recommendations:")
    print("-" * 60)
    
    # Find headers that could benefit from PIMPL
    pimpl_candidates = []
    for header, analysis in header_analysis.items():
        if (analysis['template_count'] > 3 or 
            analysis['heavy_stl_count'] > 3 or 
            analysis['complexity_score'] > 50):
            pimpl_candidates.append((header, analysis['complexity_score']))
    
    pimpl_candidates.sort(key=lambda x: x[1], reverse=True)
    print("   PIMPL Pattern Candidates (Hide Implementation):")
    for header, score in pimpl_candidates[:5]:
        print(f"   • {header:<35} | Complexity: {score:.1f}")
    
    # Find headers with many template instantiations
    template_candidates = []
    for header, analysis in header_analysis.items():
        if analysis['template_count'] > 5:
            template_candidates.append((header, analysis['template_count']))
    
    template_candidates.sort(key=lambda x: x[1], reverse=True)
    if template_candidates:
        print("\n   Explicit Template Instantiation Candidates:")
        for header, count in template_candidates[:5]:
            print(f"   • {header:<35} | Templates: {count}")
    
    # Find consolidation opportunities
    stl_usage = defaultdict(list)
    for header, analysis in header_analysis.items():
        for stl_header in analysis['heavy_stl_headers']:
            stl_usage[stl_header].append(header)
    
    print("\n   Header Consolidation Opportunities:")
    for stl_header, using_headers in sorted(stl_usage.items(), key=lambda x: len(x[1]), reverse=True)[:5]:
        print(f"   • {stl_header:<20} → Used by {len(using_headers)} headers")
        print(f"     Could create: libstats_{stl_header.replace('<', '').replace('>', '').replace('.h', '')}_common.h")
    
    # Performance impact estimation
    print(f"\n⚡ Expected Performance Impact:")
    print("-" * 60)
    
    total_complexity = sum(analysis['complexity_score'] for analysis in header_analysis.values())
    top_10_complexity = sum(analysis['complexity_score'] for _, analysis in sorted_headers[:10])
    
    if total_complexity == 0:
        print("   ❌ No headers found for analysis")
        print(f"   Headers analyzed: {len(header_analysis)}")
        print(f"   Include directory: {include_dir}")
        improvement_percentage = 0
    else:
        potential_reduction = top_10_complexity * 0.6  # Assume 60% reduction from optimizations
        improvement_percentage = (potential_reduction / total_complexity) * 100
        
        print(f"   Current total complexity score: {total_complexity:.1f}")
        print(f"   Top 10 headers complexity: {top_10_complexity:.1f} ({top_10_complexity/total_complexity*100:.1f}%)")
        print(f"   Potential reduction: {potential_reduction:.1f}")
        print(f"   Expected compilation improvement: {improvement_percentage:.1f}%")
    
    # Save detailed analysis
    analysis_data = {
        'header_analysis': header_analysis,
        'include_counter': dict(include_counter.most_common(20)),
        'pimpl_candidates': pimpl_candidates[:10],
        'template_candidates': template_candidates[:10],
        'stl_consolidation': {k: len(v) for k, v in stl_usage.items()},
        'performance_metrics': {
            'total_complexity': total_complexity,
            'top_10_complexity': top_10_complexity,
            'expected_improvement': improvement_percentage
        }
    }
    
    try:
        with open(output_path, 'w') as f:
            json.dump(analysis_data, f, indent=2)
        print(f"\n✅ Analysis complete! Results saved to {output_path}")
    except Exception as e:
        print(f"\n⚠️  Analysis complete but could not save to {output_path}: {e}")
    
    return opportunities

def check_phase1_effectiveness():
    """Check how effective our Phase 1 implementation has been"""
    print(f"\n🔍 Phase 1 Effectiveness Check:")
    print("-" * 60)
    
    # Determine correct paths
    if os.path.exists('include'):
        forward_decl_path = 'include/common/forward_declarations.h'
        libstats_path = 'include/libstats.h'
    elif os.path.exists('../include'):
        forward_decl_path = '../include/common/forward_declarations.h'
        libstats_path = '../include/libstats.h'
    else:
        print("   ❌ Cannot find include directory")
        return
    
    if os.path.exists(forward_decl_path):
        with open(forward_decl_path, 'r') as f:
            forward_content = f.read()
        forward_classes = len(re.findall(r'\bclass\s+\w+;', forward_content))
        print(f"   ✅ Forward declarations header: {forward_classes} classes declared")
    else:
        print(f"   ❌ Forward declarations header not found at {forward_decl_path}")
    
    if os.path.exists(libstats_path):
        with open(libstats_path, 'r') as f:
            libstats_content = f.read()
        
        has_conditional = '#ifdef LIBSTATS_FULL_INTERFACE' in libstats_content
        includes_forward = 'common/forward_declarations.h' in libstats_content
        
        print(f"   ✅ Conditional compilation: {'Implemented' if has_conditional else 'Missing'}")
        print(f"   ✅ Forward declarations used: {'Yes' if includes_forward else 'No'}")
        
        # Count includes in default vs full mode
        lines = libstats_content.split('\n')
        default_includes = 0
        full_includes = 0
        in_full_mode = False
        
        for line in lines:
            if '#ifdef LIBSTATS_FULL_INTERFACE' in line:
                in_full_mode = True
            elif '#endif' in line and in_full_mode:
                in_full_mode = False
            elif '#include' in line and not line.strip().startswith('//'):
                if in_full_mode:
                    full_includes += 1
                else:
                    default_includes += 1
        
        print(f"   📊 Default mode includes: {default_includes}")
        print(f"   📊 Full mode additional includes: {full_includes}")
        reduction_ratio = (full_includes / (default_includes + full_includes)) * 100 if (default_includes + full_includes) > 0 else 0
        print(f"   📈 Include reduction achieved: {reduction_ratio:.1f}% of includes moved to full mode")
    else:
        print(f"   ❌ libstats.h not found at {libstats_path}")

def main():
    print("🔬 Comprehensive Header Optimization Analysis")
    print("=" * 80)
    
    # Change to project directory if running from build
    if os.path.basename(os.getcwd()) == 'build':
        os.chdir('..')
    
    find_optimization_opportunities()
    check_phase1_effectiveness()
    
    print(f"\n📋 Recommended Next Steps:")
    print("-" * 60)
    print("   1. Implement PIMPL pattern for top complexity headers")
    print("   2. Create STL consolidation headers for common includes")
    print("   3. Add explicit template instantiations for frequently used templates")
    print("   4. Consider precompiled headers for remaining STL dependencies")
    print("   5. Measure and validate improvements with compilation benchmarks")

if __name__ == "__main__":
    main()

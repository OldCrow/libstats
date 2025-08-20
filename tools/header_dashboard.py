#!/usr/bin/env python3
"""
Header Health Dashboard - Quick Status Check
============================================

A simple dashboard showing the current health of your header optimization.
Perfect for quick status checks or regular monitoring.
"""

import subprocess
import os
import time
from pathlib import Path

def get_compiler_config():
    """Get compiler configuration."""
    compiler = '/usr/local/opt/llvm/bin/clang++' if os.path.exists('/usr/local/opt/llvm/bin/clang++') else 'clang++'
    if compiler.startswith('/usr/local/opt/llvm'):
        return [compiler, '-std=c++20', '-stdlib=libc++', '-I/usr/local/opt/llvm/include/c++/v1']
    else:
        return [compiler, '-std=c++20']

def quick_compile_test(header_name, include_dir):
    """Quick compilation test for a single header."""
    header_path = os.path.join(include_dir, header_name)
    if not os.path.exists(header_path):
        return None

    base_flags = get_compiler_config()
    cmd = base_flags + ['-E', '-I', include_dir, header_path]

    start = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        duration = time.time() - start

        if result.returncode == 0:
            lines = len(result.stdout.split('\n'))
            return {
                'status': 'ok',
                'time': duration,
                'lines': lines,
                'size_kb': len(result.stdout) / 1024
            }
        else:
            return {'status': 'error', 'error': result.stderr[:100]}
    except subprocess.TimeoutExpired:
        return {'status': 'timeout'}
    except Exception as e:
        return {'status': 'error', 'error': str(e)[:100]}

def count_headers():
    """Count headers in the project."""
    include_dir = Path(__file__).parent.parent / "include"
    headers = list(include_dir.rglob("*.h"))

    categories = {
        'common': len([h for h in headers if 'common' in str(h)]),
        'core': len([h for h in headers if 'core' in str(h)]),
        'platform': len([h for h in headers if 'platform' in str(h)]),
        'distributions': len([h for h in headers if 'distributions' in str(h)]),
        'cache': len([h for h in headers if 'cache' in str(h)]),
        'other': 0
    }

    total_categorized = sum(categories.values()) - categories['other']
    categories['other'] = len(headers) - total_categorized

    return len(headers), categories

def main():
    """Display header health dashboard."""
    print("📊 HEADER HEALTH DASHBOARD")
    print("=" * 50)

    include_dir = str(Path(__file__).parent.parent / "include")

    # Header count overview
    total_headers, categories = count_headers()
    print(f"📁 Total headers: {total_headers}")
    print("   Distribution:")
    for category, count in categories.items():
        if count > 0:
            print(f"   • {category}: {count}")
    print()

    # Quick compilation health check
    print("⚡ COMPILATION HEALTH CHECK:")
    key_headers = [
        ("libstats.h", "Main header"),
        ("common/forward_declarations.h", "Forward decls"),
        ("platform/platform_constants.h", "Platform constants"),
        ("distributions/gaussian.h", "Distribution example")
    ]

    health_score = 0
    total_tests = 0

    for header, description in key_headers:
        result = quick_compile_test(header, include_dir)
        total_tests += 1

        if result and result['status'] == 'ok':
            # Scoring criteria
            if result['time'] < 0.3:
                status = "🟢 Excellent"
                points = 4
            elif result['time'] < 0.8:
                status = "🟡 Good"
                points = 3
            elif result['time'] < 2.0:
                status = "🟠 Moderate"
                points = 2
            else:
                status = "🔴 Slow"
                points = 1

            health_score += points

            # Bloat assessment
            bloat_ratio = result['lines'] / 1000  # Per 1000 lines
            if bloat_ratio > 150:
                bloat_indicator = "❌ Heavy"
            elif bloat_ratio > 50:
                bloat_indicator = "⚠️ Medium"
            else:
                bloat_indicator = "✅ Light"

            print(f"   {status} | {bloat_indicator} | {header}")
            print(f"      {result['time']:.2f}s, {result['lines']:,} lines")

        elif result and result['status'] == 'timeout':
            print(f"   🔴 Timeout | {header}")
            print(f"      Takes >10s to process")
            health_score += 0

        else:
            print(f"   ❌ Error | {header}")
            if result and 'error' in result:
                print(f"      {result['error']}")
            health_score += 0

    # Overall health score
    max_score = total_tests * 4
    health_percentage = (health_score / max_score) * 100

    print()
    print("🎯 OVERALL HEALTH SCORE:")
    if health_percentage >= 80:
        health_emoji = "🟢"
        health_desc = "Excellent"
        recommendation = "Headers are well optimized! 🎉"
    elif health_percentage >= 60:
        health_emoji = "🟡"
        health_desc = "Good"
        recommendation = "Some optimization opportunities remain."
    elif health_percentage >= 40:
        health_emoji = "🟠"
        health_desc = "Moderate"
        recommendation = "Significant optimization needed."
    else:
        health_emoji = "🔴"
        health_desc = "Poor"
        recommendation = "Headers need major optimization work."

    print(f"   {health_emoji} {health_percentage:.0f}% - {health_desc}")
    print(f"   💡 {recommendation}")
    print()

    # Quick recommendations
    print("🚀 QUICK ACTIONS:")
    if health_percentage < 80:
        print("   • Run: python3 tools/header_insights.py")
        print("   • Focus on slowest headers first")
        print("   • Consider PIMPL pattern for heavy headers")
    else:
        print("   • Monitor weekly with this dashboard")
        print("   • Consider advanced optimizations")
        print("   • Share your success with the team!")

    # Trend indicators (placeholder for future versions)
    print()
    print("📈 OPTIMIZATION PROGRESS:")
    print("   • Phase 2 completed: Common headers consolidated ✅")
    print("   • PIMPL patterns: Ready for implementation 🔄")
    print("   • STL consolidation: Partially implemented 🔄")
    print()

    print(f"Last updated: {time.strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()

# Header Analysis Tools

Quick reference for libstats header optimization tools.

## 🚀 Quick Start (Recommended)

### Daily Health Check
```bash
python3 tools/header_dashboard.py
```
Shows overall header health score and quick status.

### Weekly Optimization Planning
```bash
python3 tools/header_insights.py
```
Detailed analysis with clear action plans and time estimates.

### Before/After Measurements
```bash
python3 tools/compilation_benchmark.py
```
Raw performance metrics for measuring optimization improvements.

## 📚 Full Documentation

See [Header Tools Guide](../docs/HEADER_TOOLS_GUIDE.md) for complete usage instructions and interpretation guide.

## 🛠️ Legacy Tools

These tools still work but are less user-friendly:

- `header_analysis.py` - Include dependency analysis
- `static_analysis.py` - Clang-based unused include detection
- `header_optimization_analysis.py` - Comprehensive scoring
- `demo_phase1_optimization.py` - Phase 1 optimization demo
- `demo_phase2_optimization.py` - Phase 2 optimization demo

## ✅ Current Status

After Phase 2 header reorganization:
- **Header Health Score**: 81% (Excellent)
- **Build Time**: ~3 minutes
- **Common Headers**: 12 headers consolidated in `include/common/`
- **Test Pass Rate**: 90% (36/40 tests passing)

## 🎯 Next Steps

1. Run `header_insights.py` for detailed optimization recommendations
2. Focus on HIGH priority items (PIMPL pattern for heavy headers)
3. Consider STL consolidation for `string`, `vector`, `cstddef`
4. Monitor weekly with `header_dashboard.py`

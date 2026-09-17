INK, BAND, CARD, GOLD, MUTED_C, MUTED_I = "#161b23", "#232a36", "#e7e0cf", "#c8a24e", "#6b6455", "#9a9382"
OK, ERR = "#6f9a5a", "#c0563f"
W = 860; out = []
def t(x, y, s, size=9.5, fill=INK, weight=None, anchor="middle", extra=""):
    w = f' font-weight="{weight}"' if weight else ""
    out.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" text-anchor="{anchor}"{w}{extra}>{s}</text>')
def box(x, y, w, h, title, lines, tsize=11):
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="7" fill="{CARD}" stroke="{GOLD}" stroke-width="1.2"/>')
    cx = x + w/2; t(cx, y+17, title, tsize, INK, "bold")
    for i, l in enumerate(lines):
        muted = l.startswith("~"); t(cx, y+31+i*12, l[1:] if muted else l, 8.5, MUTED_C if muted else INK)
def band(x, y, w, h, heading):
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{BAND}" stroke="{GOLD}" stroke-width="1.5"/>')
    t(x+14, y+18, heading, 11.5, GOLD, "bold", "start", ' letter-spacing="0.5"')
def flow(d, color=GOLD, head="flow"): out.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="1.6" marker-end="url(#{head})"/>')
def line(d): out.append(f'<path d="{d}" fill="none" stroke="{GOLD}" stroke-width="1.6"/>')
def reads(d): out.append(f'<path d="{d}" fill="none" stroke="{GOLD}" stroke-width="1.6" stroke-dasharray="5 3" marker-end="url(#del)"/>')
def label(x, y, s, anchor="middle", color=MUTED_I):
    w = len(s)*4.6+8; lx = x - (w/2 if anchor=="middle" else (0 if anchor=="start" else w))
    out.append(f'<rect x="{lx}" y="{y-9}" width="{w}" height="12" fill="{INK}"/>'); t(x, y, s, 8.5, color, None, anchor)

t(20, 30, "libstats batch dispatch pipeline", 15, GOLD, "bold", "start", ' letter-spacing="0.5"')
t(20, 46, "span-based batch call → DispatchUtils::autoDispatch → strategy selection → execution · five threshold tables · v2.4.0", 10.5, MUTED_I, None, "start")

# A user call
box(230, 60, 400, 46, "User call — span-based batch API", ["dist.getProbability(span&lt;const double&gt; values, span&lt;double&gt; results, PerformanceHint hint)"])
flow("M430 106 V136")
# B autoDispatch
box(230, 138, 400, 58, "DispatchUtils::autoDispatch&lt;D&gt;", ["validate spans (size mismatch throws · no-aliasing contract, #112) · count == 0 → early exit", "selectStrategy(hint, count)"])
flow("M430 196 V224")
# decision diamond
out.append(f'<polygon points="430,226 505,258 430,290 355,258" fill="{CARD}" stroke="{GOLD}" stroke-width="1.2"/>')
t(430, 255, "hint.strategy", 8.5, INK, "bold"); t(430, 267, "== AUTO ?", 8.5, INK)
# YES -> PerformanceDispatcher (left) ; NO -> mapHintToStrategy (right)
PDX, PDW, MHX, MHW, ROW_Y, ROW_H = 40, 390, 470, 350, 320, 116
flow(f"M355 258 H{PDX+PDW/2} V{ROW_Y-2}", OK, "flow_ok"); label(300, 252, "YES — AUTO", color=OK)
flow(f"M505 258 H{MHX+MHW/2} V{ROW_Y-2}", ERR, "flow_err"); label(575, 252, "NO — forced hint", color=ERR)
box(PDX, ROW_Y, PDW, ROW_H, "PerformanceDispatcher  (AUTO)", [
    "SIMDPolicy::getBestLevel() → table: kNeon · kAvx · kAvx2 · kAvx512 · kNone",
    "SSE2 delegates to kAvx (matching 128-bit width)",
    "threshold[dist][op] from dispatch_thresholds.h (dashed arrow → tables below)",
    "count ≥ threshold → PARALLEL, else VECTORIZED",
    "NEVER sentinel → VECTORIZED at every size (SIMD dominates, or fork loses)",
    "~thresholds measured per machine — never compare across architectures"])
box(MHX, ROW_Y, MHW, ROW_H, "mapHintToStrategy  (forced)", [
    "FORCE_SCALAR → SCALAR",
    "FORCE_VECTORIZED → VECTORIZED",
    "MINIMIZE_LATENCY → size check, small N stays serial",
    "MAXIMIZE_THROUGHPUT → OS-aware parallel choice",
    "~bypasses the threshold tables entirely"])
# merge into executeStrategy
EX_Y = 486
line(f"M{PDX+PDW/2} {ROW_Y+ROW_H} V458 H418"); flow(f"M418 458 V{EX_Y-2}")
line(f"M{MHX+MHW/2} {ROW_Y+ROW_H} V458 H442"); flow(f"M442 458 V{EX_Y-2}")
box(230, EX_Y, 400, 44, "executeStrategy(strategy, dist, values, results)", ["branch on the Strategy enum value"])
# bus to four strategies
SX = [70, 258, 446, 634]; SW, SY, SH = 176, 576, 104
line(f"M430 {EX_Y+44} V546"); line(f"M{SX[0]+SW/2} 546 H{SX[3]+SW/2}")
for x in SX: flow(f"M{x+SW/2} 546 V{SY-2}")
box(SX[0], SY, SW, SH, "SCALAR", ["for i in 0..count:", "results[i] = scalar_func(dist, values[i])", "", "~per-element libm path", "~~2–5 ns/elem"])
box(SX[1], SY, SW, SH, "VECTORIZED", ["batch_func(dist, values, results)", "→ *BatchUnsafeImpl → VectorOps", "lane width: 8d AVX-512 · 4d AVX2/AVX", "2d NEON/SSE2 · 1d scalar fallback", "~fastest for moderate N"])
box(SX[2], SY, SW, SH, "PARALLEL", ["parallel_func(dist, values, results)", "ParallelUtils::parallelForSlices (#143)", "element-denominated serial gate,", "SIMD-slice callbacks", "~GCD · std::execution::par · Win32 pool"])
box(SX[3], SY, SW, SH, "WORK_STEALING", ["work_stealing_func(…)", "GlobalWorkStealingPool::getInstance()", "→ WorkStealingPool::parallelForSlices", "", "~irregular / uneven per-element cost"])
# threshold tables band
TB_Y = 736; band(20, TB_Y, 790, 172, "THRESHOLD TABLES — dispatch_thresholds.h · Gaussian row shown: {PDF, LogPDF, CDF} · count ≥ value → PARALLEL")
tables = [("kNeon", "Apple M1 · NEON", ["PDF      50 000", "LogPDF     64", "CDF     NEVER"], "measured 2026-09-04"),
          ("kAvx", "Kaby Lake, capped build", ["PDF      50 000", "LogPDF  150 000", "CDF      4 096"], "LIBSTATS_MAX_SIMD_TIER=AVX"),
          ("kAvx2", "Kaby Lake i7-7820HQ", ["PDF     130 000", "LogPDF   20 000", "CDF      8 192"], "AVX2+FMA · 4P/8T"),
          ("kAvx512", "Ryzen 7445HS Zen 4", ["PDF   1 000 000", "LogPDF  400 000", "CDF     25 000"], "AVX-512 · 6P/12T · MSVC"),
          ("kNone", "no SIMD · scalar loop", ["PDF       8 192", "LogPDF    8 192", "CDF       8 192"], "T2 class: exp + erf")]
for i, (name, hw, rows, note) in enumerate(tables):
    x = 43 + i*152
    out.append(f'<rect x="{x}" y="{TB_Y+30}" width="142" height="108" rx="7" fill="{CARD}" stroke="{GOLD}" stroke-width="1.2"/>')
    t(x+71, TB_Y+47, name, 11, INK, "bold"); t(x+71, TB_Y+60, hw, 8.5, MUTED_C)
    for j, r in enumerate(rows): t(x+71, TB_Y+78+j*13, r, 9.5, INK, None, "middle", ' font-family="ui-monospace,Menlo,Consolas,monospace"')
    t(x+71, TB_Y+128, note, 8.5, MUTED_C)
t(34, TB_Y+160, "SSE2 → kAvx · NEVER = VECTORIZED always · values are per-machine measurements (three quiet --large runs); kNone is an unprofiled placeholder", 8.5, MUTED_I, None, "start")
# lookup arrow: PerformanceDispatcher → tables (dashed reads), down the left gutter
reads(f"M52 {ROW_Y+ROW_H} V{TB_Y-2}"); label(62, 726, "threshold lookup", "start")
# legend + footer
LG_Y = TB_Y+172+16
out.append(f'<rect x="20" y="{LG_Y}" width="330" height="78" rx="7" fill="{CARD}" stroke="{GOLD}" stroke-width="1.2"/>')
flow(f"M34 {LG_Y+16} H90"); t(100, LG_Y+20, "control / data flow", 8.5, INK, None, "start")
flow(f"M34 {LG_Y+33} H90", OK, "flow_ok"); t(100, LG_Y+37, "decision: affirmative branch", 8.5, INK, None, "start")
flow(f"M34 {LG_Y+50} H90", ERR, "flow_err"); t(100, LG_Y+54, "decision: negative branch", 8.5, INK, None, "start")
reads(f"M34 {LG_Y+67} H90"); t(100, LG_Y+71, "reads the threshold tables (dashed, open head)", 8.5, INK, None, "start")
H = LG_Y+78+26
t(W-20, H-10, "libstats v2.4.0  ·  docs/UML/dispatch-pipeline.svg", 9.5, MUTED_I, None, "end")
svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">
<title>libstats batch dispatch pipeline — v2.4.0</title>
<defs>
 <marker id="flow" markerUnits="userSpaceOnUse" markerWidth="10" markerHeight="10" refX="9" refY="5" orient="auto"><polygon points="1 1,9 5,1 9" fill="{GOLD}"/></marker>
 <marker id="flow_ok" markerUnits="userSpaceOnUse" markerWidth="10" markerHeight="10" refX="9" refY="5" orient="auto"><polygon points="1 1,9 5,1 9" fill="{OK}"/></marker>
 <marker id="flow_err" markerUnits="userSpaceOnUse" markerWidth="10" markerHeight="10" refX="9" refY="5" orient="auto"><polygon points="1 1,9 5,1 9" fill="{ERR}"/></marker>
 <marker id="del" markerUnits="userSpaceOnUse" markerWidth="12" markerHeight="12" refX="11" refY="6" orient="auto"><path d="M1 1 L11 6 L1 11" fill="none" stroke="{GOLD}" stroke-width="1.6"/></marker>
</defs>
<rect width="{W}" height="{H}" fill="{INK}"/>
<g font-family="system-ui,-apple-system,Helvetica,Arial,sans-serif">
{chr(10).join(out)}
</g>
</svg>
'''
open(__import__("os").path.join(__import__("os").path.dirname(__file__), "dispatch-pipeline.svg"),"w").write(svg); print("dispatch height", H)

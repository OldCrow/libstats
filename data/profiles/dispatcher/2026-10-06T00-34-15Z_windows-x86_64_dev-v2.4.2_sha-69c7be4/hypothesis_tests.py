"""Test the v2.5.0 dispatch-trend hypotheses against K, M and Z's freeze-head R8 captures.

Distances are in grid steps (24 sizes, NEVER one past the end). A row's run noise is the spread
of its three runs' sustained crossovers; a difference counts as real only beyond that noise.
"""
import csv, math, os, re, statistics as st
from collections import defaultdict

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."))
PROF = os.path.join(REPO, "data", "profiles", "dispatcher")
B = {
    ("K", "AVX2"): "2026-10-05T01-50-52Z_darwin-x86_64_dev-v2.4.2_sha-6f86996",
    ("K", "AVX"): "2026-10-05T03-09-27Z_darwin-x86_64_dev-v2.4.2_sha-6f86996",
    ("K", "SSE2"): "2026-10-05T04-31-33Z_darwin-x86_64_dev-v2.4.2_sha-6f86996",
    ("M", "NEON"): "2026-10-05T04-36-25Z_darwin-arm64_dev-v2.4.2_sha-2c5230a",
    ("Z", "AVX-512"): "2026-10-06T00-34-15Z_windows-x86_64_dev-v2.4.2_sha-69c7be4",
    ("Z", "AVX2"): "2026-10-06T01-04-43Z_windows-x86_64_dev-v2.4.2_sha-69c7be4",
    ("Z", "AVX"): "2026-10-06T01-34-40Z_windows-x86_64_dev-v2.4.2_sha-69c7be4",
    ("Z", "SSE2"): "2026-10-06T02-05-13Z_windows-x86_64_dev-v2.4.2_sha-69c7be4",
}
TABLE = {"AVX-512": "kAvx512", "AVX2": "kAvx2", "AVX": "kAvx", "SSE2": "kAvx", "NEON": "kNeon"}
OPS = ["PDF", "LogPDF", "CDF"]
INF = float("inf")


def load(path):
    d = defaultdict(dict)
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            d[(r["Distribution"], r["Operation"], int(r["BatchSize"]))][r["Strategy"]] = float(r["MedianTime_us"])
    return d


def sustained(d, dist, op):
    sizes = sorted(s for (dd, oo, s) in d if dd == dist and oo == op)
    ok = None
    for s in reversed(sizes):
        r = d[(dist, op, s)]
        if "PARALLEL" in r and "VECTORIZED" in r and r["PARALLEL"] < r["VECTORIZED"]:
            ok = s
        else:
            break
    return INF if ok is None else ok


def table(name):
    src = open(os.path.join(REPO, "include", "libstats", "core", "dispatch_thresholds.h"), encoding="utf-8").read()
    body = src.split(f"constexpr ArchTable {name} = {{{{", 1)[1].split("}};", 1)[0]
    t = {}
    for m in re.finditer(r"/\*\s*([A-Z_]+)\(\d+\)\s*\*/\s*\{([^}]*)\}", body):
        vals = [INF if v.strip() == "NEVER" else float(v.strip()) for v in m.group(2).split(",")]
        t[m.group(1).replace("_", "")] = dict(zip(OPS, vals))
    return t


def key(dist):
    return {"NegBinomial": "NEGATIVEBINOMIAL"}.get(dist, dist.upper())


runs = {k: [load(os.path.join(PROF, b, f"strategy_profile_run{i}.csv")) for i in (1, 2, 3)] for k, b in B.items()}
any_run = next(iter(runs.values()))[0]
GRID = sorted({s for (_, _, s) in any_run})
DISTS = sorted({d for (d, _, _) in any_run})


def idx(x):
    return len(GRID) if x == INF else min(range(len(GRID)), key=lambda i: abs(GRID[i] - x))


def median_idx(xs):
    return st.median(idx(x) for x in xs)


def vcost(rs, dist, op, n=100000):
    """VECTORIZED per-element cost (ns) at n, median of runs."""
    return st.median(r[(dist, op, n)]["VECTORIZED"] for r in rs) * 1000 / n


def pcost(rs, dist, op, n=100000):
    return st.median(r[(dist, op, n)]["PARALLEL"] for r in rs) * 1000 / n


rows = {}  # (machine, tier, dist, op) -> dict
for (mach, tier), rs in runs.items():
    tab = table(TABLE[tier])
    for dist in DISTS:
        for op in OPS:
            xs = [sustained(r, dist, op) for r in rs]
            ix = [idx(x) for x in xs]
            tv = tab.get(key(dist), {}).get(op)
            rows[(mach, tier, dist, op)] = dict(
                xs=xs, ci=st.median(ix), spread=max(ix) - min(ix),
                ti=None if tv is None else idx(tv), cost=vcost(rs, dist, op), pc=pcost(rs, dist, op))


def out(s=""):
    print(s)


def fmt(i):
    i = round(i)
    return "NEVER" if i >= len(GRID) else (f"{GRID[i]//1000}k" if GRID[i] >= 10000 else str(GRID[i]))


out("H1 - capture vs table: is the difference beyond each row's own run noise?")
out("     (beyond = |median - table| > max(1, run spread) grid steps)")
out(f"  {'set':12}{'rows':>5}{'spread>2 steps':>15}{'differ>1':>9}{'beyond noise':>13}{'of which sooner':>16}{'later':>6}")
for (mach, tier) in B:
    rr = [v for (m, t, _, _), v in rows.items() if (m, t) == (mach, tier) and v["ti"] is not None]
    wide = sum(v["spread"] > 2 for v in rr)
    differ = [v for v in rr if abs(v["ci"] - v["ti"]) > 1]
    beyond = [v for v in rr if abs(v["ci"] - v["ti"]) > max(1, v["spread"])]
    sooner = sum(v["ci"] < v["ti"] for v in beyond)
    out(f"  {mach+' '+tier:12}{len(rr):5}{wide:15}{len(differ):9}{len(beyond):13}{sooner:16}{len(beyond)-sooner:6}")

out("\nH3 - cost model: log2(crossover) against log2(VECTORIZED ns/element at n=1e5), finite crossovers above the 64 floor")
out("     expected slope -1 if crossover = overhead / cost; residual SD in grid steps vs run noise")
fits = {}
for (mach, tier) in B:
    pts = [(math.log2(v["cost"]), math.log2(GRID[round(v["ci"])])) for (m, t, _, _), v in rows.items()
           if (m, t) == (mach, tier) and 0 < round(v["ci"]) < len(GRID)]
    nev = [v["cost"] for (m, t, _, _), v in rows.items() if (m, t) == (mach, tier) and round(v["ci"]) >= len(GRID)]
    fin = [v["cost"] for (m, t, _, _), v in rows.items() if (m, t) == (mach, tier) and round(v["ci"]) < len(GRID)]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    mx, my = st.mean(xs), st.mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs); sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx; icpt = my - slope * mx
    res = [y - (icpt + slope * x) for x, y in zip(xs, ys)]
    r2 = 1 - sum(e * e for e in res) / sum((y - my) ** 2 for y in ys)
    # fixed slope -1: intercept = log2(overhead-equivalent ns)
    c1 = st.median(y + x for x, y in zip(xs, ys))
    res1 = [y - (c1 - x) for x, y in zip(xs, ys)]
    fits[(mach, tier)] = c1
    sp = st.median(v["spread"] for (m, t, _, _), v in rows.items() if (m, t) == (mach, tier))
    out(f"  {mach+' '+tier:12} n={len(pts):2} slope {slope:+.2f} R2 {r2:.2f} | slope -1: K=2^{c1:.1f} ns, "
        f"|resid| median {st.median(abs(e) for e in res1):.1f} log2 units | NEVER rows median cost {st.median(nev) if nev else float('nan'):.2f} ns vs finite {st.median(fin):.2f} ns | run spread median {sp} steps")

out("\nH2 - is AUTO reluctant (table later than measured), overall and by per-element cost tercile?")
for (mach, tier) in B:
    rr = sorted([v for (m, t, _, _), v in rows.items() if (m, t) == (mach, tier) and v["ti"] is not None],
                key=lambda v: v["cost"])
    k = len(rr) // 3
    parts = [("cheap", rr[:k]), ("mid", rr[k:2 * k]), ("costly", rr[2 * k:])]
    s = "  ".join(f"{name} late/early {sum(v['ti'] > v['ci'] + max(1, v['spread']) for v in p)}/{sum(v['ti'] < v['ci'] - max(1, v['spread']) for v in p)}"
                  for name, p in parts)
    out(f"  {mach+' '+tier:12} {s}")
out("  (late = table crosses later than measured beyond noise, i.e. AUTO reluctant)")

out("\nH4 - SSE2 vs AVX on the same machine (beyond both rows' noise)")
for mach in ("K", "Z"):
    so = la = same = 0
    for dist in DISTS:
        for op in OPS:
            a, s2 = rows[(mach, "AVX", dist, op)], rows[(mach, "SSE2", dist, op)]
            tol = max(1, a["spread"], s2["spread"])
            if s2["ci"] < a["ci"] - tol: so += 1
            elif s2["ci"] > a["ci"] + tol: la += 1
            else: same += 1
    out(f"  {mach}: SSE2 sooner {so}, later {la}, within noise {same}")
    cr = [rows[(mach, 'SSE2', d, o)]['cost'] / rows[(mach, 'AVX', d, o)]['cost'] for d in DISTS for o in OPS]
    out(f"     SSE2/AVX vectorized cost ratio: median {st.median(cr):.2f}, p10 {sorted(cr)[len(cr)//10]:.2f}, p90 {sorted(cr)[9*len(cr)//10]:.2f}")

out("\nH5 - Z vs K, same tier: real differences, and does per-element cost explain them?")
for tier in ("AVX2", "AVX", "SSE2"):
    so = la = 0; pts = []
    for dist in DISTS:
        for op in OPS:
            z, k = rows[("Z", tier, dist, op)], rows[("K", tier, dist, op)]
            tol = max(1, z["spread"], k["spread"])
            if z["ci"] < k["ci"] - tol: so += 1
            elif z["ci"] > k["ci"] + tol: la += 1
            if 0 < round(z["ci"]) < len(GRID) and 0 < round(k["ci"]) < len(GRID):
                pts.append((math.log2(k["cost"] / z["cost"]), math.log2(GRID[round(z["ci"])] / GRID[round(k["ci"])])))
    cr = st.median(2 ** p[0] for p in pts)
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    mx, my = st.mean(xs), st.mean(ys); sxx = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx else float("nan")
    out(f"  {tier:5} beyond noise: Z sooner {so}, Z later {la} | K/Z vectorized cost median {cr:.2f}x | "
        f"log2(n*Z/n*K) vs log2(cK/cZ): slope {slope:+.2f} (cost model predicts +1), mean offset {my:+.2f}")
out("  cost-model K per machine (2^x ns, slope -1 fits): " + ", ".join(f"{m} {t} {c:.1f}" for (m, t), c in fits.items()))

out("\nHolistic - one model n* = 2^(K_machine) / cost: fraction of all rows within run noise (+1 step)")
tot = ok = 0
for (mach, tier, dist, op), v in rows.items():
    if v["cost"] <= 0: continue
    pred = 2 ** (fits[(mach, tier)]) / v["cost"]
    pi = len(GRID) if pred > GRID[-1] * 1.5 else idx(pred)
    tot += 1; ok += abs(pi - v["ci"]) <= max(1, v["spread"]) + 1
out(f"  model within noise+1 step: {ok}/{tot} ({100*ok/tot:.0f}%)")
tot = ok = 0
for (mach, tier, dist, op), v in rows.items():
    if v["ti"] is None: continue
    tot += 1; ok += abs(v["ti"] - v["ci"]) <= max(1, v["spread"]) + 1
out(f"  current tables within noise+1 step: {ok}/{tot} ({100*ok/tot:.0f}%)")

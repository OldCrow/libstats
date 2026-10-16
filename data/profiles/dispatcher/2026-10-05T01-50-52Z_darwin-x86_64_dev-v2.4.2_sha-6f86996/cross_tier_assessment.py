#!/usr/bin/env python3
"""Cross-tier assessment of the 2026-10-05 Kaby Lake captures (AVX2, AVX, SSE2) at 6f86996
(code at the freeze head 5367999; later commits are docs only).

Sustained V->P crossover per (dist, op, run) as in the 2026-09-04 bundles; candidate per row =
median of the three runs, or the max (conservative) when max/min > 2 (NEVER = inf).
Compares candidates to the current kAvx2 / kAvx tables and to the same tier's
2026-10-04 capture at d9384f8 (before the speed work b5a47df-5367999).
"""
import csv, re, statistics
from collections import defaultdict

import os
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."))
PROF = f"{REPO}/data/profiles/dispatcher"
NEWD = {"AVX2": f"{PROF}/2026-10-05T01-50-52Z_darwin-x86_64_dev-v2.4.2_sha-6f86996",
        "AVX": f"{PROF}/2026-10-05T03-09-27Z_darwin-x86_64_dev-v2.4.2_sha-6f86996",
        "SSE2": f"{PROF}/2026-10-05T04-31-33Z_darwin-x86_64_dev-v2.4.2_sha-6f86996"}
OLDD = {"AVX2": f"{PROF}/2026-10-04T02-07-22Z_darwin-x86_64_dev-v2.4.2_sha-d9384f8",
        "AVX": f"{PROF}/2026-10-04T03-51-16Z_darwin-x86_64_dev-v2.4.2_sha-d9384f8",
        "SSE2": f"{PROF}/2026-10-04T05-16-48Z_darwin-x86_64_dev-v2.4.2_sha-d9384f8"}
INF = float("inf")
OPS = ["PDF", "LogPDF", "CDF"]

def load(path):
    d = defaultdict(dict)
    for r in csv.DictReader(open(path)):
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

def candidate(xs):
    lo, hi = min(xs), max(xs)
    if hi == INF and lo == INF:
        return INF
    if lo == 0 or hi / lo > 2:
        return hi
    return statistics.median(xs)

def fmt(x):
    return "NEVER" if x == INF else (f"{x/1000:g}k" if x >= 10000 else f"{x:g}")

def table(name):
    src = open(f"{REPO}/include/libstats/core/dispatch_thresholds.h").read()
    body = src.split(f"constexpr ArchTable {name} = {{{{", 1)[1].split("}};", 1)[0]
    t = {}
    for m in re.finditer(r"/\*\s*([A-Z_]+)\(\d+\)\s*\*/\s*\{([^}]*)\}", body):
        vals = [INF if v.strip() == "NEVER" else float(v.strip()) for v in m.group(2).split(",")]
        t[m.group(1).replace("_", "")] = dict(zip(OPS, vals))
    return t

def key(dist):
    return {"NegBinomial": "NEGATIVEBINOMIAL"}.get(dist, dist.upper())

runs = {t: [load(f"{NEWD[t]}/strategy_profile_run{i}.csv") for i in (1, 2, 3)] for t in ("AVX2", "AVX", "SSE2")}
old = {t: [load(f"{OLDD[t]}/strategy_profile_run{i}.csv") for i in (1, 2, 3)] for t in OLDD}
dists = sorted({dd for (dd, oo, s) in runs["AVX2"][0]})
grid = sorted({s for (_, _, s) in runs["AVX2"][0]})

def steps(a, b):  # grid-step distance, NEVER one past the end
    idx = lambda x: len(grid) if x == INF else min(range(len(grid)), key=lambda i: abs(grid[i] - x))
    return abs(idx(a) - idx(b))

cand = {}
for t in runs:
    for dist in dists:
        for op in OPS:
            xs = [sustained(r, dist, op) for r in runs[t]]
            cand[(t, dist, op)] = (candidate(xs), xs)

for t, tab in (("AVX2", "kAvx2"), ("AVX", "kAvx"), ("SSE2", "kAvx")):
    cur = table(tab)
    print(f"\n== {t}: candidate vs current {tab} (rows > 1 grid step apart)")
    n = 0
    for dist in dists:
        for op in OPS:
            c, xs = cand[(t, dist, op)]
            now = cur[key(dist)][op]
            if steps(c, now) > 1:
                n += 1
                oldxs = ""
                if t in old:
                    oxs = [sustained(r, dist, op) for r in old[t]]
                    oldxs = f"  d9384f8 runs {'/'.join(fmt(x) for x in oxs)}"
                print(f"  {dist:15} {op:6} table {fmt(now):>6} -> cand {fmt(c):>6}   runs {'/'.join(fmt(x) for x in xs)}{oldxs}")
    print(f"  {n} of {len(dists)*len(OPS)} rows would change")

print("\n== speed work: candidate vs the same tier at d9384f8 (rows > 1 grid step apart)")
ocand = {(t, d, o): candidate([sustained(r, d, o) for r in old[t]]) for t in old for d in dists for o in OPS}
for t in runs:
    moved = [(d, o) for d in dists for o in OPS if steps(cand[(t, d, o)][0], ocand[(t, d, o)]) > 1]
    print(f"  {t}: {len(moved)} of {len(dists)*len(OPS)}")
    for d, o in moved:
        print(f"    {d:15} {o:6} d9384f8 {fmt(ocand[(t, d, o)]):>6} -> {fmt(cand[(t, d, o)][0]):>6}   runs {'/'.join(fmt(x) for x in cand[(t, d, o)][1])}")

print("\n== run-to-run: rows whose three new runs span > 2x (candidate takes the max)")
for t in runs:
    wide = [(d, o) for d in dists for o in OPS
            if (lambda xs: min(xs) != max(xs) and (min(xs) == 0 or max(xs) / min(xs) > 2))(cand[(t, d, o)][1])]
    print(f"  {t}: {len(wide)}  " + ", ".join(f"{d}.{o}" for d, o in wide))

print("\n== SSE2 vs AVX candidates (SSE2 delegates to kAvx today): rows > 1 grid step apart")
diff = [(d, o) for d in dists for o in OPS if steps(cand[("SSE2", d, o)][0], cand[("AVX", d, o)][0]) > 1]
for d, o in diff:
    print(f"  {d:15} {o:6} AVX {fmt(cand[('AVX', d, o)][0]):>6}  SSE2 {fmt(cand[('SSE2', d, o)][0]):>6}")
print(f"  {len(diff)} of {len(dists)*len(OPS)}")

print("\n== tier ordering of candidates (expect SSE2 <= AVX <= AVX2: slower VECTORIZED -> parallel sooner)")
viol = [(d, o) for d in dists for o in OPS
        if not (cand[("SSE2", d, o)][0] <= cand[("AVX", d, o)][0] * 2 and cand[("AVX", d, o)][0] <= cand[("AVX2", d, o)][0] * 2)]
for d, o in viol:
    print(f"  {d:15} {o:6} " + "  ".join(f"{t} {fmt(cand[(t, d, o)][0])}" for t in ("SSE2", "AVX", "AVX2")))
print(f"  {len(viol)} rows out of order by > 2x")

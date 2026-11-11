#!/usr/bin/env python3
"""Cross-tier assessment of the 2026-10-13/14 Zen 4 clang-cl captures (AVX-512, AVX2, AVX, SSE2)
at d1adc95 (the freeze head after the fix-9 round; branch tip df78931).

Sustained V->P crossover per (dist, op, run) as in the 2026-09-04 bundles; candidate per row =
median of the three runs, or the max (conservative) when max/min > 2 (NEVER = inf).
Compares candidates to the current kAvx512 / kAvx2 / kAvx tables (SSE2 delegates to kAvx), to
Z's previous capture of the same tier (2026-10-07, 8271f6e: code ee11e62, before #191 and every
fix round since), and to K's 2026-10-13 capture of the same tier at d1adc95 (AVX2, AVX, SSE2).
"""
import csv, re, statistics
from collections import defaultdict

import os
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."))
PROF = f"{REPO}/data/profiles/dispatcher"
NEWD = {"AVX-512": f"{PROF}/2026-10-14T02-41-05Z_windows-x86_64_dev-v2.4.2_sha-df78931",
        "AVX2": f"{PROF}/2026-10-14T03-05-23Z_windows-x86_64_dev-v2.4.2_sha-df78931",
        "AVX": f"{PROF}/2026-10-14T03-29-48Z_windows-x86_64_dev-v2.4.2_sha-df78931",
        "SSE2": f"{PROF}/2026-10-14T03-54-51Z_windows-x86_64_dev-v2.4.2_sha-df78931"}
OLDD = {"AVX-512": f"{PROF}/2026-10-07T02-10-06Z_windows-x86_64_dev-v2.4.2_sha-8271f6e",
        "AVX2": f"{PROF}/2026-10-07T02-35-21Z_windows-x86_64_dev-v2.4.2_sha-8271f6e",
        "AVX": f"{PROF}/2026-10-07T02-59-39Z_windows-x86_64_dev-v2.4.2_sha-8271f6e",
        "SSE2": f"{PROF}/2026-10-07T03-24-20Z_windows-x86_64_dev-v2.4.2_sha-8271f6e"}
KD = {"AVX2": f"{PROF}/2026-10-13T13-11-55Z_darwin-x86_64_dev-v2.4.2_sha-d1adc95",
      "AVX": f"{PROF}/2026-10-13T13-58-00Z_darwin-x86_64_dev-v2.4.2_sha-d1adc95",
      "SSE2": f"{PROF}/2026-10-13T14-44-55Z_darwin-x86_64_dev-v2.4.2_sha-d1adc95"}
TABLES = (("AVX-512", "kAvx512"), ("AVX2", "kAvx2"), ("AVX", "kAvx"), ("SSE2", "kAvx"))
TIERS = [t for t, _ in TABLES]
INF = float("inf")
OPS = ["PDF", "LogPDF", "CDF"]

def load(path):
    d = defaultdict(dict)
    for r in csv.DictReader(open(path, encoding="utf-8", newline="")):
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
    src = open(f"{REPO}/include/libstats/core/dispatch_thresholds.h", encoding="utf-8").read()
    body = src.split(f"constexpr ArchTable {name} = {{{{", 1)[1].split("}};", 1)[0]
    t = {}
    for m in re.finditer(r"/\*\s*([A-Z_]+)\(\d+\)\s*\*/\s*\{([^}]*)\}", body):
        vals = [INF if v.strip() == "NEVER" else float(v.strip()) for v in m.group(2).split(",")]
        t[m.group(1).replace("_", "")] = dict(zip(OPS, vals))
    return t

def key(dist):
    return {"NegBinomial": "NEGATIVEBINOMIAL"}.get(dist, dist.upper())

runs = {t: [load(f"{NEWD[t]}/strategy_profile_run{i}.csv") for i in (1, 2, 3)] for t in TIERS}
old = {t: [load(f"{OLDD[t]}/strategy_profile_run{i}.csv") for i in (1, 2, 3)] for t in OLDD}
kruns = {t: [load(f"{KD[t]}/strategy_profile_run{i}.csv") for i in (1, 2, 3)] for t in KD}
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
ocand = {(t, d, o): candidate([sustained(r, d, o) for r in old[t]]) for t in old for d in dists for o in OPS}
kcand = {(t, d, o): candidate([sustained(r, d, o) for r in kruns[t]]) for t in kruns for d in dists for o in OPS}

for t, tab in TABLES:
    cur = table(tab)
    print(f"\n== {t}: candidate vs current {tab} (rows > 1 grid step apart)")
    n = 0
    for dist in dists:
        for op in OPS:
            c, xs = cand[(t, dist, op)]
            now = cur[key(dist)][op]
            if steps(c, now) > 1:
                n += 1
                extra = f"  8271f6e {fmt(ocand[(t, dist, op)]):>6}"
                if t in kcand:
                    extra += f"  K {fmt(kcand[(t, dist, op)]):>6}"
                print(f"  {dist:15} {op:6} table {fmt(now):>6} -> cand {fmt(c):>6}   runs {'/'.join(fmt(x) for x in xs)}{extra}")
    print(f"  {n} of {len(dists)*len(OPS)} rows would change")

print("\n== since Z's last capture: candidate vs the same tier at 8271f6e (code ee11e62; rows > 1 grid step apart)")
for t in TIERS:
    moved = [(d, o) for d in dists for o in OPS if steps(cand[(t, d, o)][0], ocand[(t, d, o)]) > 1]
    print(f"  {t}: {len(moved)} of {len(dists)*len(OPS)}")
    for d, o in moved:
        print(f"    {d:15} {o:6} 8271f6e {fmt(ocand[(t, d, o)]):>6} -> {fmt(cand[(t, d, o)][0]):>6}   runs {'/'.join(fmt(x) for x in cand[(t, d, o)][1])}")

print("\n== Z vs K at d1adc95, same tier (rows > 1 grid step apart; K = Kaby Lake 4C/8T AppleClang 15)")
for t in kruns:
    diff = [(d, o) for d in dists for o in OPS if steps(cand[(t, d, o)][0], kcand[(t, d, o)]) > 1]
    print(f"  {t}: {len(diff)} of {len(dists)*len(OPS)}")
    for d, o in diff:
        print(f"    {d:15} {o:6} Z {fmt(cand[(t, d, o)][0]):>6}  K {fmt(kcand[(t, d, o)]):>6}")

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

print("\n== tier ordering of candidates (expect SSE2 <= AVX <= AVX2 <= AVX-512: slower VECTORIZED -> parallel sooner)")
order = ("SSE2", "AVX", "AVX2", "AVX-512")
viol = [(d, o) for d in dists for o in OPS
        if not all(cand[(order[i], d, o)][0] <= cand[(order[i + 1], d, o)][0] * 2 for i in range(len(order) - 1))]
for d, o in viol:
    print(f"  {d:15} {op:6} " if False else f"  {d:15} {o:6} " + "  ".join(f"{t} {fmt(cand[(t, d, o)][0])}" for t in order))
print(f"  {len(viol)} rows out of order by > 2x")

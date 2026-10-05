#!/usr/bin/env python3
"""Assessment of the 2026-10-05 M1 NEON captures at 2c5230a (dev/v2.4.2 freeze head plus tests/docs).

Method as K's 2026-10-04 cross_tier_assessment.py: sustained V->P crossover per (dist, op, run);
candidate = median of the three runs, or the max when max/min > 2 (NEVER = inf). Compares
candidates to the current kNeon table and to the 2026-09-04 v2.4.0 NEON runs.
"""
import csv, re, statistics, os
from collections import defaultdict

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."))
PROF = f"{REPO}/data/profiles/dispatcher"
NEWD = {"NEON": f"{PROF}/2026-10-05T04-36-25Z_darwin-arm64_dev-v2.4.2_sha-2c5230a"}
OLDD = {"NEON": f"{PROF}/2026-09-04T04-22-28Z_darwin-arm64_dev-v2.4.0_sha-5f27ee1"}
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

runs = {t: [load(f"{NEWD[t]}/strategy_profile_run{i}.csv") for i in (1, 2, 3)] for t in NEWD}
old = {t: [load(f"{OLDD[t]}/strategy_profile_run{i}.csv") for i in (1, 2, 3)] for t in OLDD}
dists = sorted({dd for (dd, oo, s) in runs["NEON"][0]})
grid = sorted({s for (_, _, s) in runs["NEON"][0]})

def steps(a, b):  # grid-step distance, NEVER one past the end
    idx = lambda x: len(grid) if x == INF else min(range(len(grid)), key=lambda i: abs(grid[i] - x))
    return abs(idx(a) - idx(b))

cand = {}
for t in runs:
    for dist in dists:
        for op in OPS:
            xs = [sustained(r, dist, op) for r in runs[t]]
            cand[(t, dist, op)] = (candidate(xs), xs)
oldc = {(d, o): candidate([sustained(r, d, o) for r in old["NEON"]]) for d in dists for o in OPS}

cur = table("kNeon")
print("== NEON: candidate vs current kNeon (rows > 1 grid step apart)")
n = moved = 0
for dist in dists:
    for op in OPS:
        c, xs = cand[("NEON", dist, op)]
        now = cur[key(dist)][op]
        if steps(c, now) > 1:
            n += 1
            oxs = [sustained(r, dist, op) for r in old["NEON"]]
            mv = steps(c, oldc[(dist, op)]) > 1
            moved += mv
            print(f"  {dist:15} {op:6} table {fmt(now):>6} -> cand {fmt(c):>6}   runs {'/'.join(fmt(x) for x in xs)}"
                  f"  Sep runs {'/'.join(fmt(x) for x in oxs)}  {'MOVED since Sep' if mv else 'same as Sep'}")
print(f"  {n} of {len(dists)*len(OPS)} rows would change; {moved} of them moved since the 2026-09-04 capture")

print("\n== NEON: candidates moved > 1 grid step since 2026-09-04 (v2.4.0 -> v2.4.2), any table agreement")
m = [(d, o) for d in dists for o in OPS if steps(cand[("NEON", d, o)][0], oldc[(d, o)]) > 1]
for d, o in m:
    print(f"  {d:15} {o:6} Sep {fmt(oldc[(d, o)]):>6} -> now {fmt(cand[('NEON', d, o)][0]):>6}   table {fmt(cur[key(d)][o]):>6}")
print(f"  {len(m)} of {len(dists)*len(OPS)}")

print("\n== run-to-run: rows whose three new runs span > 2x (candidate takes the max)")
wide = [(d, o) for d in dists for o in OPS
        if (lambda xs: min(xs) != max(xs) and (min(xs) == 0 or max(xs) / min(xs) > 2))(cand[("NEON", d, o)][1])]
print(f"  NEON: {len(wide)}  " + ", ".join(f"{d}.{o}" for d, o in wide))

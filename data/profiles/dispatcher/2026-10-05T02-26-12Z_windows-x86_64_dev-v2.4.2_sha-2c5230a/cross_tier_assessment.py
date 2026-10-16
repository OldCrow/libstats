#!/usr/bin/env python3
"""Cross-tier assessment of the 2026-10-05 Zen 4 captures (AVX-512, AVX2, AVX, SSE2) at 2c5230a.

Adapted from the Kaby Lake AVX2 bundle's script (2026-10-04T02-07-22Z). Sustained V->P crossover
per (dist, op, run); candidate per row = median of the three runs, or the max (conservative) when
max/min > 2 (NEVER = inf). Compares candidates to the current tables (AVX-512 -> kAvx512, AVX2 ->
kAvx2, AVX and SSE2 -> kAvx) and to Kaby Lake's same-tier captures at d9384f8. Those K captures
predate the 2026-10-04 speed work (b5a47df-5367999), so a Z-vs-K difference mixes machine and code.
"""
import csv, os, re, statistics
from collections import defaultdict

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."))
PROF = f"{REPO}/data/profiles/dispatcher"
NEWD = {"AVX-512": f"{PROF}/2026-10-05T02-26-12Z_windows-x86_64_dev-v2.4.2_sha-2c5230a",
        "AVX2": f"{PROF}/2026-10-05T03-07-38Z_windows-x86_64_dev-v2.4.2_sha-2c5230a",
        "AVX": f"{PROF}/2026-10-05T03-50-00Z_windows-x86_64_dev-v2.4.2_sha-2c5230a",
        "SSE2": f"{PROF}/2026-10-05T04-33-12Z_windows-x86_64_dev-v2.4.2_sha-2c5230a"}
KD = {"AVX2": f"{PROF}/2026-10-04T02-07-22Z_darwin-x86_64_dev-v2.4.2_sha-d9384f8",
      "AVX": f"{PROF}/2026-10-04T03-51-16Z_darwin-x86_64_dev-v2.4.2_sha-d9384f8",
      "SSE2": f"{PROF}/2026-10-04T05-16-48Z_darwin-x86_64_dev-v2.4.2_sha-d9384f8"}
TABLE = {"AVX-512": "kAvx512", "AVX2": "kAvx2", "AVX": "kAvx", "SSE2": "kAvx"}
TIERS = ["AVX-512", "AVX2", "AVX", "SSE2"]
INF = float("inf")
OPS = ["PDF", "LogPDF", "CDF"]


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
    with open(f"{REPO}/include/libstats/core/dispatch_thresholds.h", encoding="utf-8") as f:
        src = f.read()
    body = src.split(f"constexpr ArchTable {name} = {{{{", 1)[1].split("}};", 1)[0]
    t = {}
    for m in re.finditer(r"/\*\s*([A-Z_]+)\(\d+\)\s*\*/\s*\{([^}]*)\}", body):
        vals = [INF if v.strip() == "NEVER" else float(v.strip()) for v in m.group(2).split(",")]
        t[m.group(1).replace("_", "")] = dict(zip(OPS, vals))
    return t


def key(dist):
    return {"NegBinomial": "NEGATIVEBINOMIAL"}.get(dist, dist.upper())


runs = {t: [load(f"{NEWD[t]}/strategy_profile_run{i}.csv") for i in (1, 2, 3)] for t in TIERS}
krun = {t: [load(f"{KD[t]}/strategy_profile_run{i}.csv") for i in (1, 2, 3)] for t in KD}
dists = sorted({dd for (dd, oo, s) in runs["AVX-512"][0]})
grid = sorted({s for (_, _, s) in runs["AVX-512"][0]})
N = len(dists) * len(OPS)


def steps(a, b):  # grid-step distance, NEVER one past the end
    idx = lambda x: len(grid) if x == INF else min(range(len(grid)), key=lambda i: abs(grid[i] - x))
    return abs(idx(a) - idx(b))


def cands(rs):
    out = {}
    for dist in dists:
        for op in OPS:
            xs = [sustained(r, dist, op) for r in rs]
            out[(dist, op)] = (candidate(xs), xs)
    return out


cand = {t: cands(runs[t]) for t in TIERS}
kcand = {t: cands(krun[t]) for t in KD}

for t in TIERS:
    tab = TABLE[t]
    cur = table(tab)
    print(f"\n== {t}: Z candidate vs current {tab} (rows > 1 grid step apart)")
    n = n_k = 0
    for dist in dists:
        for op in OPS:
            c, xs = cand[t][(dist, op)]
            now = cur.get(key(dist), {}).get(op)
            if now is None:
                continue
            if steps(c, now) > 1:
                n += 1
                kx = ""
                if t in kcand:
                    kc = kcand[t][(dist, op)][0]
                    agree = steps(c, kc) <= 1
                    n_k += agree
                    kx = f"  K {fmt(kc):>6}{'  (K agrees)' if agree else ''}"
                print(f"  {dist:15} {op:6} table {fmt(now):>6} -> cand {fmt(c):>6}   runs {'/'.join(fmt(x) for x in xs)}{kx}")
    tail = f"; K's d9384f8 candidate within one step of Z's on {n_k} of them" if t in kcand else ""
    print(f"  {n} of {N} rows would change{tail}")

print("\n== Z vs K, same tier (candidates > 1 grid step apart; K at d9384f8, before the speed work)")
for t in KD:
    diff = [(d, o) for d in dists for o in OPS if steps(cand[t][(d, o)][0], kcand[t][(d, o)][0]) > 1]
    print(f"  {t}: {len(diff)} of {N}")
    for d, o in diff:
        print(f"    {d:15} {o:6} Z {fmt(cand[t][(d, o)][0]):>6}  K {fmt(kcand[t][(d, o)][0]):>6}")

print("\n== run-to-run: rows whose three runs span > 2x (candidate takes the max)")
for t in TIERS:
    wide = [(d, o) for d in dists for o in OPS
            if (lambda xs: min(xs) != max(xs) and (min(xs) == 0 or max(xs) / min(xs) > 2))(cand[t][(d, o)][1])]
    print(f"  {t}: {len(wide)}  " + ", ".join(f"{d}.{o}" for d, o in wide))

print("\n== SSE2 vs AVX candidates (SSE2 delegates to kAvx today): rows > 1 grid step apart")
diff = [(d, o) for d in dists for o in OPS if steps(cand["SSE2"][(d, o)][0], cand["AVX"][(d, o)][0]) > 1]
for d, o in diff:
    print(f"  {d:15} {o:6} AVX {fmt(cand['AVX'][(d, o)][0]):>6}  SSE2 {fmt(cand['SSE2'][(d, o)][0]):>6}")
print(f"  {len(diff)} of {N} (K: 18)")

print("\n== tier ordering (expect SSE2 <= AVX <= AVX2 <= AVX-512 within 2x: slower VECTORIZED -> parallel sooner)")
order = ["SSE2", "AVX", "AVX2", "AVX-512"]
viol = [(d, o) for d in dists for o in OPS
        if any(cand[a][(d, o)][0] > cand[b][(d, o)][0] * 2 for a, b in zip(order, order[1:]))]
for d, o in viol:
    print(f"  {d:15} {o:6} " + "  ".join(f"{t} {fmt(cand[t][(d, o)][0])}" for t in order))
print(f"  {len(viol)} rows out of order by > 2x")

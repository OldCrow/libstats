#!/usr/bin/env python3
"""Sustained V->P crossover extraction from strategy_profile CSVs.

Sustained crossover: smallest batch size where PARALLEL median beats
VECTORIZED median at that size AND every larger size in the grid.
(NOT the first-crossing heuristic - it reports 64-element noise ties.)
"""
import csv, sys
from collections import defaultdict

DIR = "/private/tmp/claude-501/-Users-wolfman-Development-corvus/a8f9fd77-3c3c-4dde-a2c1-b38fb1d3dee2/scratchpad/kaby_leg"

def load(path):
    d = defaultdict(dict)  # (dist,op,size) -> {strategy: time}
    with open(path) as f:
        for row in csv.DictReader(f):
            d[(row["Distribution"], row["Operation"], int(row["BatchSize"]))][row["Strategy"]] = float(row["MedianTime_us"])
    return d

def sustained(d, dist, op):
    sizes = sorted(s for (dd, oo, s) in d if dd == dist and oo == op)
    if not sizes:
        return None
    cross = "NEVER"
    # walk from largest down; sustained = longest suffix where P < V
    ok_from = None
    for s in reversed(sizes):
        r = d[(dist, op, s)]
        if "PARALLEL" not in r or "VECTORIZED" not in r:
            break
        if r["PARALLEL"] < r["VECTORIZED"]:
            ok_from = s
        else:
            break
    return ok_from if ok_from is not None else "NEVER"

runs = [load(f"{DIR}/strategy_profile_run{i}.csv") for i in (1, 2, 3)]
dists = sorted({dd for (dd, oo, s) in runs[0]})
ops = ["PDF", "LogPDF", "CDF"]

print(f"{'Distribution':<18}{'Op':<8}{'run1':>10}{'run2':>10}{'run3':>10}")
for dist in dists:
    for op in ops:
        vals = [sustained(r, dist, op) for r in runs]
        if all(v is None for v in vals):
            continue
        print(f"{dist:<18}{op:<8}" + "".join(f"{str(v):>10}" for v in vals))

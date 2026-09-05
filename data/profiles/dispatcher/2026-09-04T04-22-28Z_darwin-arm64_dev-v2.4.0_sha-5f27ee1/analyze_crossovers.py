#!/usr/bin/env python3
"""Sustained V->P crossover analysis for strategy_profile CSVs.

Methodology (PR #143, Zen 4 leg): the threshold for a (distribution, op)
is the smallest measured batch size S such that PARALLEL beats VECTORIZED
at S and at EVERY larger measured size (sustained win), not the
validator's first-crossing heuristic. If VECTORIZED (or SCALAR) is best
at the largest size, the row is NEVER.

Usage: sustained_crossover.py run1.csv [run2.csv ...]
Prints per-run sustained crossovers and best-at-max strategy per row.
"""
import csv, sys
from collections import defaultdict

NEVER = "NEVER"

def load(path):
    d = defaultdict(dict)  # (dist, op) -> {size: {strategy: time}}
    with open(path) as f:
        for row in csv.DictReader(f):
            key = (row["Distribution"], row["Operation"])
            size = int(row["BatchSize"])
            d[key].setdefault(size, {})[row["Strategy"]] = float(row["MedianTime_us"])
    return d

def sustained(sizes_map):
    sizes = sorted(sizes_map)
    maxsize = sizes[-1]
    # best strategy at max size
    at_max = sizes_map[maxsize]
    best_at_max = min(at_max, key=at_max.get)
    # sustained: smallest S where PARALLEL < VECTORIZED for S and all larger
    cross = NEVER
    for i, s in enumerate(sizes):
        ok = True
        for t in sizes[i:]:
            m = sizes_map[t]
            if "PARALLEL" not in m or "VECTORIZED" not in m or m["PARALLEL"] >= m["VECTORIZED"]:
                ok = False
                break
        if ok:
            cross = s
            break
    return cross, best_at_max, at_max

def main():
    runs = [load(p) for p in sys.argv[1:]]
    keys = sorted(set().union(*[set(r) for r in runs]))
    hdr = "  ".join(f"run{i+1}" for i in range(len(runs)))
    print(f"{'Distribution':<20} {'Op':<8} {hdr:<24} best@max (per run)")
    for key in keys:
        cells, bests = [], []
        for r in runs:
            if key not in r:
                cells.append("-"); bests.append("-"); continue
            cross, best, _ = sustained(r[key])
            cells.append(str(cross)); bests.append(best[:1])  # S/V/P/W
        print(f"{key[0]:<20} {key[1]:<8} {'  '.join(f'{c:<8}' for c in cells)}  {'/'.join(bests)}")

if __name__ == "__main__":
    main()

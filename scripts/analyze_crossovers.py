#!/usr/bin/env python3
"""Canonical sustained-crossover analysis for dispatcher profiling (#146).

Input: `strategy_profile --large -o <csv>` files (columns Distribution,
Operation, BatchSize, Strategy, MedianTime_us), normally the three quiet runs
of one bundle. Python 3 standard library only.

Method (scripts/PROFILING_METHOD.md, 2026-09-04 amendment):

  * Per run, the SUSTAINED V->P crossover of a (distribution, op) is the
    smallest grid size where PARALLEL beats VECTORIZED at that size AND at
    every larger grid size; NEVER if there is none. (Not the first-crossing
    heuristic, which reports 64-element noise ties.)
  * Candidate across runs: the median, or the max (conservative) when
    max/min > 2. NEVER counts as infinity, so a NEVER run forces NEVER
    whenever it is the median or the spread rule applies. With an even run
    count the median is the upper middle value (conservative).
  * Resolution clamp. dispatch_thresholds.h: timing resolution floors at
    ~0.1-0.2 us, so a crossover derived from sizes at or below the grid floor
    (64) is a noise artifact, to be clamped upward to at least 64 rather than
    encoded as measured. On a grid that starts at 64 the upward clamp alone
    changes no value, so this tool applies the reading the kAvx/kAvx2
    comments show in use ("64 is resolution noise ... conservative"): a run
    whose sustained crossover sits at the grid floor is flagged, kept in the
    per-run table, and enters the candidate at the floor, where the 2x rule
    lets any larger run win as max. A candidate that equals the floor is
    flagged `floor` (unresolved: reads "parallel from the smallest size", not
    a measured crossover); never below the floor.

Usage:
  analyze_crossovers.py --bundle DIR [--table kAvx2]
  analyze_crossovers.py run1.csv run2.csv run3.csv [--table kNeon]
  analyze_crossovers.py --self-test

--table parses that ArchTable from include/libstats/core/dispatch_thresholds.h
and flags each row `ok` when the candidate is within one grid step of the
encoded value, else `DIFF`. A row whose table line carries a comment is also
marked `override?`: many encoded rows are deliberate overrides backed by other
evidence recorded in the comment.
"""
import argparse
import csv
import glob
import os
import re
import sys
from collections import defaultdict

NEVER = "NEVER"
INF = float("inf")
OPS = ["PDF", "LogPDF", "CDF"]
SPREAD_LIMIT = 2.0
FLOOR = 64
HEADER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                      "include", "libstats", "core", "dispatch_thresholds.h")


def load(path):
    """-> {(dist, op): {size: {strategy: median_us}}}"""
    d = defaultdict(lambda: defaultdict(dict))
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            d[(row["Distribution"], row["Operation"])][int(row["BatchSize"])][
                row["Strategy"]] = float(row["MedianTime_us"])
    return d


def sustained(sizes_map):
    """Smallest size where PARALLEL < VECTORIZED at it and every larger size."""
    result = NEVER
    for s in sorted(sizes_map, reverse=True):
        m = sizes_map[s]
        if "PARALLEL" in m and "VECTORIZED" in m and m["PARALLEL"] < m["VECTORIZED"]:
            result = s
        else:
            break
    return result


def candidate(values):
    """Median of runs, or max when max/min > 2 (NEVER = inf). -> int | NEVER"""
    v = sorted(INF if x == NEVER else x for x in values)
    lo, hi = v[0], v[-1]
    if hi == INF or hi / lo > SPREAD_LIMIT:
        pick = hi
    else:
        pick = v[len(v) // 2]
    return NEVER if pick == INF else int(max(pick, FLOOR))


def grid_pos(grid, x):
    """Fractional grid index of x (linear between neighbours), clamped to ends."""
    if x <= grid[0]:
        return 0.0
    if x >= grid[-1]:
        return float(len(grid) - 1)
    for i in range(len(grid) - 1):
        if grid[i] <= x <= grid[i + 1]:
            return i + (x - grid[i]) / (grid[i + 1] - grid[i])
    return float(len(grid) - 1)


def within_one_step(grid, a, b):
    if a == NEVER or b == NEVER:
        return a == b
    return abs(grid_pos(grid, a) - grid_pos(grid, b)) <= 1.0 + 1e-9


def norm(name):
    return re.sub(r"[^a-z0-9]", "", name.lower())


def parse_table(header_path, table):
    """-> {norm_dist: ((pdf, log_pdf, cdf), has_comment)} for ArchTable `table`."""
    with open(header_path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    start = next((i for i, l in enumerate(lines)
                  if re.match(r"\s*constexpr ArchTable %s\b" % re.escape(table), l)), None)
    if start is None:
        sys.exit("table %s not found in %s" % (table, header_path))
    rows, cur = {}, None
    row_re = re.compile(r"/\*\s*([A-Z_0-9]+?)\((\d+)\)\s*\*/\s*\{([^}]*)\}(.*)")
    for l in lines[start + 1:]:
        if l.strip().startswith("}};"):
            break
        m = row_re.search(l)
        if m:
            vals = []
            for tok in m.group(3).split(","):
                tok = tok.strip()
                vals.append(NEVER if tok == "NEVER" else int(tok))
            cur = norm(m.group(1))
            rows[cur] = [tuple(vals), "//" in m.group(4)]
        elif cur and l.strip().startswith("//"):
            # continuation of the previous row's trailing comment (it is indented
            # past the code column); a comment at the margin belongs to a group.
            if len(l) - len(l.lstrip()) >= 40:
                rows[cur][1] = True
            else:
                cur = None
        else:
            cur = None
    return {k: (v[0], v[1]) for k, v in rows.items()}


def analyze(runs):
    """-> (rows, grid); rows = [(dist, op, per_run_values, candidate)]"""
    grid = sorted({s for r in runs for k in r for s in r[k]})
    dists = sorted({k[0] for r in runs for k in r})
    rows = []
    for dist in dists:
        for op in OPS:
            vals = [sustained(r[(dist, op)]) if (dist, op) in r else None for r in runs]
            if all(v is None for v in vals):
                continue
            present = [v for v in vals if v is not None]
            rows.append((dist, op, vals, candidate(present)))
    return rows, grid


def render(rows, grid, nruns, table=None):
    out = []
    head = "%-18s%-8s" % ("Distribution", "Op") + "".join("%10s" % ("run%d" % (i + 1)) for i in range(nruns))
    head += "%11s" % "candidate"
    if table:
        head += "  %-10s %-9s %s" % ("encoded", "match", "note")
    out.append(head)
    floor = grid[0]
    counts = defaultdict(int)
    for dist, op, vals, cand in rows:
        line = "%-18s%-8s" % (dist, op) + "".join("%10s" % ("-" if v is None else v) for v in vals)
        line += "%11s" % cand
        notes = []
        if any(v == floor for v in vals):
            notes.append("floor-run")
        if cand == floor:
            notes.append("floor")
        if table:
            tcell = table.get(norm(dist))
            if tcell is None:
                line += "  %-10s %-9s %s" % ("?", "n/a", "not in table")
                counts["n/a"] += 1
            else:
                enc = tcell[0][OPS.index(op)]
                ok = within_one_step(grid, cand, enc)
                tag = "ok" if ok else "DIFF"
                if tcell[1]:
                    notes.append("override?")
                counts[tag] += 1
                counts[tag + ("/comment" if tcell[1] else "/plain")] += 1
                line += "  %-10s %-9s %s" % (enc, tag, " ".join(notes))
                out.append(line.rstrip())
                continue
        out.append((line + "  " + " ".join(notes)).rstrip() if notes else line.rstrip())
    if table:
        out.append("")
        out.append("rows: ok=%d DIFF=%d (DIFF with table comment=%d, without=%d; ok with comment=%d)" % (
            counts["ok"], counts["DIFF"], counts["DIFF/comment"], counts["DIFF/plain"],
            counts["ok/comment"]))
    return "\n".join(out)


def self_test():
    def mk(**per_size):
        # keys "sN" -> size N; values (VECTORIZED, PARALLEL) times
        return {int(k[1:]): {"VECTORIZED": v, "PARALLEL": p} for k, (v, p) in per_size.items()}

    # sustained: transient early win must not count; suffix start wins
    assert sustained(mk(s64=(1, 0.5), s128=(1, 2), s256=(1, 2), s512=(2, 1), s1024=(3, 1))) == 512
    # win at every size -> smallest grid size
    assert sustained(mk(s64=(2, 1), s128=(2, 1))) == 64
    # NEVER: parallel loses at the largest size
    assert sustained(mk(s64=(2, 1), s128=(1, 2))) == NEVER
    # ties do not count as a win
    assert sustained(mk(s64=(1, 1), s128=(2, 1))) == 128
    # candidate: median when within 2x, max when beyond, NEVER = inf
    assert candidate([4096, 2048, 4096]) == 4096
    assert candidate([1000, 1500, 2000]) == 1500
    assert candidate([1000, 1500, 2001]) == 2001
    assert candidate([1000, 3000, 1500]) == 3000
    assert candidate([NEVER, NEVER, NEVER]) == NEVER
    assert candidate([4096, NEVER, 4096]) == NEVER
    assert candidate([NEVER, NEVER, 4096]) == NEVER
    # clamp: floor sample is flagged; 64 vs 4096 spread picks max; all-floor stays 64
    assert candidate([64, 64, 4096]) == 4096
    assert candidate([64, 64, 64]) == 64
    assert candidate([64, 96, 80]) == 80  # never below the floor either way
    assert candidate([10, 10, 10]) == FLOOR
    # table comparison: one grid step
    grid = [64, 128, 256, 512]
    assert within_one_step(grid, 128, 256) and not within_one_step(grid, 64, 256)
    assert within_one_step(grid, NEVER, NEVER) and not within_one_step(grid, 64, NEVER)
    # end to end render on synthetic rows
    rows = [("X", "PDF", [128, 256, 128], candidate([128, 256, 128]))]
    assert rows[0][3] == 128
    assert "floor" in render([("Y", "CDF", [64, 64, 64], 64)], grid, 3)
    print("self-test: ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="*", help="strategy_profile --large CSVs (one per run)")
    ap.add_argument("--bundle", help="bundle dir; reads strategy_profile_run*.csv")
    ap.add_argument("--table", help="compare against ArchTable (kAvx2|kAvx|kAvx512|kNeon|kNone)")
    ap.add_argument("--header", default=HEADER, help="path to dispatch_thresholds.h")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return
    files = list(a.csv)
    if a.bundle:
        files += sorted(glob.glob(os.path.join(a.bundle, "strategy_profile_run*.csv")))
    if not files:
        ap.error("give CSV files or --bundle DIR")
    runs = [load(p) for p in files]
    rows, grid = analyze(runs)
    table = parse_table(a.header, a.table) if a.table else None
    print(render(rows, grid, len(runs), table))


if __name__ == "__main__":
    main()

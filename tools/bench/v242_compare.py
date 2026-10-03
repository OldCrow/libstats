#!/usr/bin/env python3
"""Join two v242_cost_bench CSVs (v2.4.1, v2.4.2) and report what moved.

Usage: v242_compare.py OLD.csv NEW.csv [--threshold 1.25]

Prints three sections:
  1. Per case, the worst slowdown and best speedup over all rows (new/old).
  2. Every row whose new/old ratio is beyond the threshold either way.
  3. Dispatch: each (case, op, n) where AUTO is slower than the best forced strategy by more
     than the threshold, under either version. A row marked NEW is fine in v2.4.1 and off in
     v2.4.2, so a dispatch_thresholds.h row may need re-deriving with strategy_profile; one
     marked OLD was already off and is not a v2.4.2 change.
"""

import argparse
import csv
from collections import defaultdict


def load(path):
    rows = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows[(r["case"], r["op"], int(r["n"]), r["strategy"])] = float(r["ns_per_element"])
    return rows


def auto_gap(rows, case, op, n):
    """AUTO's time over the best forced strategy's, or None if a column is missing."""
    forced = [rows.get((case, op, n, s)) for s in ("scalar", "vectorized", "parallel")]
    auto = rows.get((case, op, n, "auto"))
    if auto is None or None in forced:
        return None
    best = min(forced)
    names = ("scalar", "vectorized", "parallel")
    return auto / best, names[forced.index(best)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--threshold", type=float, default=1.25)
    a = ap.parse_args()
    old, new = load(a.old), load(a.new)
    t = a.threshold
    keys = [k for k in new if k in old]
    missing = len(new) + len(old) - 2 * len(keys)
    if missing:
        print(f"note: {missing} rows present in only one file")

    print("== 1. per case: worst slowdown / best speedup (new/old)")
    per_case = defaultdict(list)
    for k in keys:
        per_case[k[0]].append((new[k] / old[k], k))
    for case, items in per_case.items():
        worst = max(items)
        best = min(items)
        print(f"{case:28s} worst {worst[0]:6.2f}x ({worst[1][1]} n={worst[1][2]} "
              f"{worst[1][3]})   best {best[0]:6.2f}x ({best[1][1]} n={best[1][2]} {best[1][3]})")

    print(f"\n== 2. rows beyond {t:.2f}x either way (new/old)")
    moved = sorted(((new[k] / old[k], k) for k in keys
                    if new[k] / old[k] >= t or new[k] / old[k] <= 1 / t), reverse=True)
    for ratio, (case, op, n, s) in moved:
        print(f"{ratio:7.2f}x  {case:28s} {op:12s} n={n:<7d} {s:10s} "
              f"{old[(case, op, n, s)]:9.1f} -> {new[(case, op, n, s)]:9.1f} ns")
    if not moved:
        print("none")

    print(f"\n== 3. dispatch: AUTO more than {t:.2f}x slower than the best forced strategy")
    cells = sorted({(c, o, n) for (c, o, n, s) in keys if s == "auto"})
    flagged = False
    for case, op, n in cells:
        g_new = auto_gap(new, case, op, n)
        g_old = auto_gap(old, case, op, n)
        if g_new is None or g_old is None:
            continue
        off_new, off_old = g_new[0] > t, g_old[0] > t
        if not (off_new or off_old):
            continue
        flagged = True
        mark = "NEW" if off_new and not off_old else ("OLD" if off_old and not off_new else "BOTH")
        print(f"{mark:4s} {case:28s} {op:6s} n={n:<7d} v2.4.1 auto/best {g_old[0]:5.2f} "
              f"({g_old[1]})   v2.4.2 auto/best {g_new[0]:5.2f} ({g_new[1]})")
    if not flagged:
        print("none")


if __name__ == "__main__":
    main()

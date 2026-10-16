#!/usr/bin/env python3
"""AUTO under the current vs proposed table, all distributions: per (dist, op, n) the AUTO ratio and
each version's AUTO/best-forced, with the pass-to-pass noise floor from the forced strategies."""
import csv, statistics, sys, collections
import os; R = os.environ['R']
def load(p): return {(r['case'], r['op'], int(r['n']), r['strategy']): float(r['ns_per_element']) for r in csv.DictReader(open(p))}
passes = int(sys.argv[1]) if len(sys.argv) > 1 else 2
old = [load(f"{R}/tab_old_{i}.csv") for i in range(1, passes + 1)]
new = [load(f"{R}/tab_new_{i}.csv") for i in range(1, passes + 1)]
def med(ds, k): return statistics.median(d[k] for d in ds if k in d)
keys = [k for k in old[0] if all(k in d for d in old + new)]
forced = sorted(med(new, k) / med(old, k) for k in keys if k[3] != 'auto')
print(f"noise floor (forced, same code): median {statistics.median(forced):.3f} p05 {forced[len(forced)//20]:.2f} p95 {forced[-len(forced)//20]:.2f} n={len(forced)}")
if passes > 1:
    rep = sorted(old[1][k] / old[0][k] for k in keys)
    print(f"pass-to-pass (old table, all rows): median {statistics.median(rep):.3f} p05 {rep[len(rep)//20]:.2f} p95 {rep[-len(rep)//20]:.2f}")
def best(ds, c, o, n): return min((med(ds, (c, o, n, s)), s) for s in ('vectorized', 'parallel'))
auto = [k for k in keys if k[3] == 'auto']
rows = []
for k in auto:
    c, o, n, _ = k
    oa, na = med(old, k), med(new, k)
    bo, so = best(old, c, o, n); bn, sn = best(new, c, o, n)
    rows.append((c, o, n, oa, na, na / oa, oa / bo, so, na / bn, sn))
r = sorted(x[5] for x in rows)
print(f"AUTO rows {len(rows)}: new/old median {statistics.median(r):.3f}; faster >1.25x {sum(x < 0.8 for x in r)}; slower >1.25x {sum(x > 1.25 for x in r)}")
print(f"AUTO > 1.25x behind best forced: old table {sum(x[6] > 1.25 for x in rows)}, new table {sum(x[8] > 1.25 for x in rows)}")
print("\n== REGRESSIONS: new AUTO slower than old by > 1.25x")
for x in sorted((x for x in rows if x[5] > 1.25), key=lambda x: -x[5]):
    print(f"  {x[0]:16}{x[1]:7}{x[2]:>7}  old {x[3]:8.2f} new {x[4]:8.2f}  new/old {x[5]:5.2f}  old a/b {x[6]:4.2f}({x[7][:4]}) new a/b {x[8]:4.2f}({x[9][:4]})")
print("\n== remaining gaps under the NEW table: AUTO > 1.25x behind best forced")
for x in sorted((x for x in rows if x[8] > 1.25), key=lambda x: -x[8]):
    print(f"  {x[0]:16}{x[1]:7}{x[2]:>7}  auto/best {x[8]:4.2f} (best {x[9]})  old auto/best {x[6]:4.2f}  new/old {x[5]:4.2f}")
print("\n== gaps CLOSED by the new table (old a/b > 1.25, new a/b <= 1.25)")
closed = [x for x in rows if x[6] > 1.25 and x[8] <= 1.25]
by = collections.defaultdict(list)
for x in closed: by[(x[0], x[1])].append(f"{x[2]}:{x[5]:.2f}")
for (c, o), v in sorted(by.items()): print(f"  {c:16}{o:7} n:new/old  {' '.join(v)}")
print("\n== per-row verdict for every changed proposal row would need the table; see rows_kNeon.txt")

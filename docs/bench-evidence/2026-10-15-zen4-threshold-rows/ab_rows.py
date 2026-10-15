"""Per-pass detail from a table A/B round: for each (dist, op) print, per n, old/new AUTO per pass and the
forced VECTORIZED / PARALLEL medians, plus the first n where PARALLEL beats VECTORIZED and stays ahead."""
import csv, statistics, sys
R = sys.argv[1]
want = [tuple(a.split(".")) for a in sys.argv[2:]]  # Dist.op
def load(p): return {(r["case"], r["op"], int(r["n"]), r["strategy"]): float(r["ns_per_element"]) for r in csv.DictReader(open(p, encoding="utf-8", newline=""))}
old = [load(f"{R}/tab_old_{i}.csv") for i in (1, 2)]
new = [load(f"{R}/tab_new_{i}.csv") for i in (1, 2)]
allp = old + new
def med(ds, k): return statistics.median(d[k] for d in ds if k in d)
sizes = sorted({k[2] for k in old[0]})
for dist, op in want:
    print(f"\n{dist} {op}   (ns/element; V and P are medians of the four passes)")
    print(f"  {'n':>7} {'V':>7} {'P':>7} {'P/V':>5} | old auto p1/p2 {'':>6} new auto p1/p2 {'':>6} new a/b")
    first = None
    for n in sizes:
        kv, kp, ka = (dist, op, n, "vectorized"), (dist, op, n, "parallel"), (dist, op, n, "auto")
        if ka not in old[0]: continue
        v, p = med(allp, kv), med(allp, kp)
        best = min(v, p); na = statistics.median([new[0][ka], new[1][ka]])
        if p < v and first is None: first = n
        if p >= v: first = None
        print(f"  {n:>7} {v:7.2f} {p:7.2f} {p/v:5.2f} | {old[0][ka]:7.2f} {old[1][ka]:7.2f}        {new[0][ka]:7.2f} {new[1][ka]:7.2f}        {na/best:5.2f}")
    print(f"  sustained P<V from: {first if first else 'NEVER (within 500k)'}")
